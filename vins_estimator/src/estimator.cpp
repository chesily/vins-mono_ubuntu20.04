#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}  //这里使用花括号初始化，使得若存在类型转换且具有丢失信息的风险时，编译器将会报错，即更加安全
{
    ROS_INFO("init begins");
    clearState();
}

// 设置估计器的参数，包括外参、视觉测量残差的信息矩阵和时间戳延迟
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    // 设置视觉重投影约束的信息矩阵的平方根
    // 这里可以看到虚拟相机的用法
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

// 清空或初始化滑动窗口中所有的状态量
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * @brief   处理IMU数据
 * @Description 进行IMU预积分，并使用IMU测量中值积分来优化滑窗中关键帧的状态
 * @param[in]   dt 两帧IMU之间的时间间隔
 * @param[in]   linear_acceleration 线加速度
 * @param[in]   angular_velocity 角速度
 * @return  void
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 滑窗中保留11帧，frame_count表示现在处理第几帧，一般处理到第11帧时就保持不变了
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // 由于预积分是帧间约束，因此第1个预积分量实际上是用不到的，所以只有大于0才处理
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 保存IMU测量时间差和IMU测量
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 对IMU测量逐帧中值积分来更新滑窗中系统状态量，给非线性优化提供更可信的初始值
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief   处理图像特征数据
 * @Description addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧               
 *              判断并进行外参标定
 *              进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param[in]   image 某帧图像所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   header 某帧图像的头信息
 * @return  void
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    // Step 1 检查滑窗中次新帧是否是关键帧并制定相应的边缘化策略
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        // 如果次新帧是关键帧，则滑窗中最老的帧就要被移出滑窗
        marginalization_flag = MARGIN_OLD;
    else
        // 否则移除次新帧
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // 将当前帧图像数据、时间戳、临时预积分值打包为图像帧对象
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe)); // 保存所有图像帧对象
    // 更新临时预积分
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // Step 2： 旋转外参初始化
    // 如果没有外参初值
    if(ESTIMATE_EXTRINSIC == 2)
    {
        // 标定imu和相机的旋转外参初值
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 得到次新图像帧与最新图像帧之间对应特征点（归一化坐标）
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                // 标志位设置成不准确的外参初值（注意此处并没有标定平移外参，初始化为0）
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE) // 有足够的帧数
        {
            bool result = false;
            // 要有外参，且距离上次初始化不成功至少相邻0.1s
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                // Step 3： VIO初始化
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                solver_flag = NON_LINEAR;
                // Step 4： 非线性优化求解VIO
                solveOdometry();
                // Step 5： 滑动窗口
                slideWindow();
                // Step 6： 移除无效地图点
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];  // 滑窗里最新的位姿
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0]; // 滑窗里最老的位姿
                last_P0 = Ps[0];
                
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // 检测VIO是否正常(如果异常，重启VIO)
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        // 给可视化用的
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * @brief VIO初始化，将滑窗中的P V Q恢复到参考帧并且和重力对齐
 * 
 * @return  bool true:初始化成功
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
     // STEP1：检查IMU可观性（其实可以完全注释）
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 从第二帧开始检查IMU预积分
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // “加速度预积分量”累加
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // 所有图像帧“平均加速度预积分”

        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 求"加速度预积分”标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);

        // 实际上检查结果并没有用
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enough!");
            //return false;
        }
    }

    // STEP2：求解滑动窗口中图像帧相对与参考帧（滑窗中某帧）的位姿，并三角化特征点生成地图点
     //将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中
    Quaterniond Q[frame_count + 1]; // 滑窗中图像帧到参考帧的旋转
    Vector3d T[frame_count + 1]; // 滑窗中图像帧到参考帧的平移
    map<int, Vector3d> sfm_tracked_points;  // 三角化成功的特征点的3D坐标
    vector<SFMFeature> sfm_f; // 保存每个特征点的信息用于接下来的纯视觉sfm
    // 遍历所有特征点
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1; // 这个变量和IMU无关，存储观测到特征点的图像帧的索引
        SFMFeature tmp_feature; 
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            // 观测到该特征点的图像帧的索引以及对应的相机归一化平面的坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 

    Matrix3d relative_R; // 最新帧到第l帧图像的旋转变换
    Vector3d relative_T; // 最新帧到第l帧图像的平移变换
    int l;  // 满足初始化条件（平均视差、内点数均大于给定阈值）的滑窗中参考帧

    // 判断滑窗中是否存在一帧图像与最新帧图像具有足够的视差和匹配点对，并且通过本质矩阵分解出相对位姿后的内点数大于一定的阈值
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    //对窗口中每个图像帧求解sfm问题
    //得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T(滑窗内所有帧->l帧的位姿)和特征点坐标sfm_tracked_points
    /* sfm.construct 函数本身就很巨大，里面很多步骤，代码就不贴了，大致如下：
       1. 三角化l 帧与frame_num - 1帧，得到一些地图点
       2. 通过这些地图点，利用pnp的方法计算l+1, l+2, l+3, …… frame_num-2 相对于l的位姿。而且每计算一帧，就会与frame_num - 1帧进行三角化，得出更多地图点
       3. 三角化l+1, l+2 …… frame_num-2帧与l帧
       4. 对l-1, l-2, l-3 等帧与sfm_f的特征点队列进行pnp求解，得出相对于l的位姿，并三角化其与l帧
       5. 三角化剩余的点
       6. ceres全局BA优化（仅优化位姿），最小化3d投影误差
     */
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        // 求解失败则边缘化滑窗中最老的图像帧
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // STEP3: 求解不在滑窗中图像帧的位姿
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec()) //若该帧是滑窗内的图像帧
        {
            frame_it->second.is_key_frame = true; //将该帧设为关键帧
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // 获得该帧IMU坐标系到参考帧（第l帧）的相对位姿
            frame_it->second.T = T[i]; // 初始化不估计平移外参（此时平移是从该帧相机坐标系到参考帧（第l帧）的相对平移）
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }

        // 用比该图像帧更早的最近关键帧的位姿作为pnp的初始值
        // cv::solvePnP需要的位姿是T_ck_l(T_ck_w)
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        //获取pnp需要用到的存储每个特征点三维点和图像坐标的vector
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历这一图像帧对应的特征点
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            // 由于是单目，这里id_pts.second大小就是1（camera_id）
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2)); // 地图点的世界坐标
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }

        // 构造相机内参，因为已经是归一化后坐标了，这里的内参就是单位矩阵了
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
        // 保证特征点数大于5   
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }

        /** 
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)   
         */
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        //R_ck_l
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        //这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        //R_l_ck
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        //T_l_ck
        T_pnp = R_pnp * (-T_pnp);
        //R_l_b = R_l_ck * R_c_b
        frame_it->second.R = R_pnp * RIC[0].transpose(); //当前帧的imu_k->l帧相机系
        frame_it->second.T = T_pnp; // 当前帧相机系->l帧相机系
    }

    // Step 4 视觉惯性对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate  
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x; // 视觉惯性对齐过程中待优化变量，包括窗口中每帧的速度V[0:n]、重力g、尺度s

    //solve scale
    //计算陀螺仪偏置，尺度，重力加速度和各个图像帧的速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // 保存滑窗中关键帧的位姿
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true; // TODO ？？？之前不是已经设置了吗
    }

    //将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    //重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // 三角化所有特征点（此时尺度仍是模糊的）
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    // 将滑窗中的图像帧重新预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]); // 不是更新临时预积分量
    }

    // 下面开始把所有的状态对齐到第0帧的imu坐标系（恢复尺度）
    for (int i = frame_count; i >= 0; i--)
        // 之前的Ps[i]是当前帧相机系到参考帧相机系的平移，Rs[i]是当前帧IMU系到参考帧相机系的旋转
        // twi - tw0 = toi,就是把所有的平移对齐到滑窗中的第0帧（从滑窗在第0帧到第i帧的平移在参考帧相机系下的表示）
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]); 

    // 保存滑窗中关键帧的速度（从当前imu坐标系到参考相机系）
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    // 把尺度模糊的3d点恢复到真实尺度下
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // 所有的P V Q全部对齐到第0帧的，同时和对齐到重力方向
    Matrix3d R0 = Utility::g2R(g); // 参考系（第l帧）到世界系（东北天）的旋转矩阵（不考虑偏航）R^w_l
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();  // Rs[0]是R_l_0,计算世界系到滑窗中第1帧的偏航角
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 滑窗中第一帧yaw赋0
    // 重力方向、滑窗内关键帧的位姿和速度对齐到世界系（世界系中滑窗中第一帧的偏航角为0）
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];  
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @Description    判断每帧到滑窗最新帧对应特征点的平均视差是否大于30
                   solveRelativeRT()通过基础矩阵计算最新帧图像与第l帧图像之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧（最新帧）到第l帧（参考帧）之间的旋转矩阵R
 * @param[out]   relative_T 当前帧（最新帧）到第l帧（参考帧）之间的平移向量T
 * @param[out]   l 保存滑窗中与最新帧图像满足初始化条件的那一帧图像索引
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // 得到滑窗中第i帧图像与最新帧图像之间对应特征点（归一化坐标）
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        // 特征点匹配对数需大于20
        if (corres.size() > 20)
        {
            // 计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                // 第j对匹配的特征点在滑窗中第i帧图像与最新帧图像的归一化平面坐标（x,y）
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());

            //判断是否满足初始化条件：视差>30和内点数满足要求
            //同时返回滑窗最新帧（当前帧）到第l帧（参考帧）的Rt
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

//三角化求解所有特征点的深度，并进行非线性优化
void Estimator::solveOdometry()
{
    // 保证滑窗中帧数满了
    if (frame_count < WINDOW_SIZE)
        return;
    // 其次要求初始化完成
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        //把之前还没有三角化的特征点三角化（初始化完成后也会检测到新的特征点）
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        // 非线性优化
        optimization();
    }
}

// Eigen向量转换成c++的double数组（因为ceres使用数值数组）
void Estimator::vector2double()
{
    // 滑窗中关键帧的状态转换（包括位姿、速度、IMU偏置）
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        // 以x，y，z，w的方式存储是为了便于Eigen运算
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }

    // IMU和相机之间的外参转换
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // 特征点的逆深度转换
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    // 相机和IMU之间的时间戳延迟转换
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// c++的double数组转换成Eigen向量（因为ceres使用数值数组）,并把滑窗中第一帧的偏航和位置固定
void Estimator::double2vector()
{
    // 取出优化前滑窗中第一帧的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    // ceres优化后滑窗中第一帧的姿态
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    
    // 求得优化前后的偏航角差值
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    // 俯仰角接近90度会有奇异性问题
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // 保证滑窗中第一帧的偏航角不变
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        // 保证滑窗中第一帧的位移不变
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * @brief   基于滑动窗口紧耦合的非线性优化，残差项的构造和求解
 * @Description 添加要优化的变量 (p,v,q,ba,bg) 一共15个自由度，IMU的外参也可以加进来
 *              添加残差，残差项分为4块 先验残差+IMU残差+视觉残差+闭环检测残差
 *              根据倒数第二帧是不是关键帧确定边缘化的结果           
 * @return      void
*/
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);  // 鲁棒核函数使用柯西核

    // STEP1：定义待优化的参数块，类似于g2o的顶点

    // 参数块1: 滑窗中的位姿、速度、IMU偏置（共11帧）
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 派生类的指针可以隐含转换为基类的指针（在替代之后，派生类对象就可以作为基类的对象使用，但只能使用从基类继承的成员，如果基类成员函数没有声明为虚函数，仍然会调用基类的函数）
        // 由于姿态不满足正常的加法（即李群上没有加法），因此需要自己定义他的加法
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); //Ps、Rs转变成para_Pose，7自由度
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS); //Vs、Bas、Bgs转变成para_SpeedBias,9自由度
    }

    // 参数块2：相机与IMU的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        // 如果已经有外参初始值则将外参固定
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    // 参数块3：相机与IMU的时间戳延迟
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    //给待优化变量赋初值
    vector2double();

    // STEP2：通过残差约束来添加残差块，类似于g2o的边

    // 约束1：先验约束(上一次边缘化的结果)
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // 约束2：IMU预积分约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        // IMU预积分间隔过长就不可信了
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        // 从第1帧的预积分开始（对应与第0帧和第1帧之间的IMU测量）
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        // loss_function为NULL,即没有使用鲁棒核函数（IMU优化时去除外点的方式和视觉不同）
        // 4个优化变量，维度分别是7， 9， 7， 9
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    // 约束3：视觉重投影约束
    int f_m_cnt = 0; // 用于优化的视觉测量的总个数（约束边的数目）
    int feature_index = -1;  // 有效特征点的索引
    // 遍历滑窗中的每一个特征点
    for (auto &it_per_id : f_manager.feature)
    {
        // 特征点有效性检查（被观测次数大于2，第一次被观测到的关键帧ID小于8）
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 这个特征点第一次被观测到时的相机归一化平面坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历所有能看到这特征点的关键帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j) // 自己与自己无法形成重投影
            {
                continue;
            }
            // 取出该特征点在另一帧的相机归一化平面坐标
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                    // 估计传感器之间的时延时构建的重投影残差
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                // 构建普通的重投影残差
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                // 使用柯西鲁棒核函数，待优化变量包括两个观测到该特征点的关键帧的位姿（一定包括第一次观测到该特征点的关键帧），IMU与相机的外参，该特征点的逆深度
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    // 约束4：视觉回环约束
    //添加闭环检测残差，计算滑动窗口中与每一个闭环关键帧的相对位姿，这个相对位置是为后面的图优化准备
    // 1.在回环约束中已知回环帧匹配到了哪些地图点，以id的形式记录。
    // 2.在滑动窗口内搜索具有相同id的地图点，建立重投影误差，当前帧的位姿作为初始位姿。
    // 3.优化重投影误差，得到回环帧在当前滑动窗口具有的地图点下，应该所处的位姿T_w2_relo (T_w2_bi)。
    // 这里相当于一个pnp的优化算法，用重投影误差构建优化项解算，而不是SVD解算
    // 输入是当前帧（滑窗内）的3d点,和回环帧的2d点，设当前世界为w2,即优化变量为T_w2_bi，但是这里不只优化T_w2_bi，还优化了相关的T_w2_bj
    // 其实通过这种方式修正的位姿，只利用了一帧约束，并不是非常准确，所以称之为快速重定位
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);  // 需要优化的回环帧位姿
        int retrive_feature_index = 0;
        int feature_index = -1; // 有效特征点的索引
        // 遍历滑窗中的每一个特征点
        for (auto &it_per_id : f_manager.feature)
        {
            // 特征点有效性检查（被观测次数大于2，第一次被观测到的关键帧ID小于8）
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                // 找到匹配点
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                // 在这个特征点的起始帧和回环帧之间构建重投影约束
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }
    }

    // STEP3: ceres优化求解
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    // 边缘化老的关键帧操作比较多，因此优化时间就少一些
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary); // ceres优化求解
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();

    // STEP4: 边缘化
    TicToc t_whole_marginalization;
    //case1: 如果次新帧是关键帧，将边缘化滑窗中最老帧及其看到的路标点和IMU数据，将其转化为先验
    if (marginalization_flag == MARGIN_OLD)
    {
        // 一个用来边缘化操作的对象（类似与cere的Problem）
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // 关于边缘化有几点注意的地方
        // 1、找到需要边缘化的参数块，这里是地图点，第0帧位姿，第0帧速度零偏
        // 2、找到构造高斯牛顿下降时跟这些待边缘化相关的参数块有关的残差约束，那就是预积分约束，重投影约束，以及上一次边缘化约束
        // 3、这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏

        // STEP4.1:将上一次边缘化约束添加到marginalization_info中
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            // last_marginalization_parameter_blocks是上一次边缘化对哪些当前参数块有约束
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                // 涉及到的待边缘化的上一次边缘化留下来的当前参数块只有位姿和速度零偏
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // STEP4.2：将第0帧和第1帧间的IMU预积分约束添加到marginalization_info中
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 跟构建ceres约束问题一样，这里也需要得到残差和雅克比矩阵
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); // 第0帧的位姿和IMU速度及偏置被边缘化（第0个和第1个参数块）
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // STEP4.3：将第0帧观测到的所有特征对应的视觉重投影约束添加到marginalization_info中
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature) // 遍历所有特征
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                // 只处理能被第0帧观测到的特征
                if (imu_i != 0)
                    continue;

                // 该特征在第0帧下的归一化坐标
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                // 遍历观测到该特征的所有关键帧，并建立与第0帧的视觉重投影约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    // 该特征在该关键帧下的归一化坐标
                    Vector3d pts_j = it_per_frame.point;
                    // 根据是否优化传感器之间的时间延迟确定残差矩阵
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3}); // 第0帧的位姿和该特征点的逆深度被边缘化（第0个和第3个参数块）
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        // STEP4.4：计算每一个约束块的残差和雅克比矩阵
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        // STEP4.5：使用边缘化约束更新边缘化因子残差和雅克比矩阵
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        // STEP4.6：更新滑动窗口，即将滑窗，因此记录新地址对应的老地址
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 位姿和速度都要滑窗移动
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        // 外参和时间延时不变
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        // parameter_blocks实际上就是addr_shift的索引的集合及搬进去的新地址
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;   // 本次边缘化的所有信息
        last_marginalization_parameter_blocks = parameter_blocks; // 代表该次边缘化对某些参数块形成约束，这些参数块在滑窗之后的地址
        
    }
    // case2:如果次新帧不是关键帧，则边缘化次新帧
    else 
    {
         // 要求有上一次边缘化的结果同时，即将被margin掉的在上一次边缘化后的约束中
         // 预积分结果合并，因此只有位姿margin掉
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
            // 保留次新帧的IMU测量，丢弃该帧的视觉测量，将上一次先验残差项传递给marginalization_info
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    // 速度零偏只会margin第1个，不可能出现倒数第二个
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    // 这种case只会margin掉倒数第二个位姿
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                // 这里只会更新一下边缘化因子
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            // 这里的操作如出一辙
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            // 去掉次新帧
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE) // 滑窗，最新帧成为次新帧
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else   // 其他不变
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

/**
 * @brief   实现滑动窗口all_image_frame的函数
 * @Description 如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
                如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
 * @return      void
*/
void Estimator::slideWindow()
{
    TicToc t_margin;
    // 根据边缘化种类的不同，进行滑窗的方式也不同
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        // 必须是填满了滑窗才可以
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++) //依次把滑窗内信息前移
            {
                 // 一帧一帧交换过去
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            //把滑窗末尾(10帧)信息给最新一帧(11帧)
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            // 预积分量就得置零
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            // 清空最新帧的IMU数据buffer
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

             // 删除滑窗中最老帧对应的全部信息
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);  //删掉all_image_frame内最老帧以前的所有帧(不包括最老帧)
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// 滑动窗口边缘化次新帧时处理特征点被观测的帧号
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// 滑动窗口边缘化最老帧时处理特征点被观测的帧号
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

