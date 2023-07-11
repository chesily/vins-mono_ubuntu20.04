#include "feature_manager.h"

// 返回检测到当前特征点的最新图像帧在滑窗中的编号
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

// 特征点管理器的构造函数（使用滑动窗口中的关键帧姿态数组初始化）
FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs) // 这里初始化的是指针，因此后面也会随之更新
{
    // 旋转外参初始化为单位阵
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

// 设置旋转外参
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

//清除特征点管理器中的特征点
void FeatureManager::clearState()
{
    feature.clear();
}

/**
 * @brief 获取有效的特征点的数目
 */
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        // 该特征点被至少两帧图像跟踪并且该特征点第一次被检测到时的图像帧在滑窗中的ID小于8
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief   将最新帧的特征点信息存入特征点列表feature中，并检查滑窗中次新帧是否是关键帧
 * @param[in]   frame_count 滑动窗口内图像帧的个数
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   td IMU和cam同步时间差
 * @return  bool true：滑窗中次新帧是关键帧;false：非关键帧
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum   = 0;  // 所有特征点视差总和
    int    parallax_num   = 0;  // 计算视差的次数
           last_track_num = 0;  // 当前图像帧上被跟踪到的特征点数目
    // STEP 1 将最新帧的特征点信息存入特征点列表feature中
    //把image map中的所有特征点放入feature列表容器中
    for (auto &id_pts : image)   // 遍历所有特征点 
    {
        // 特征点对象（包括对应于当前图像帧的归一化坐标、像素坐标、在归一化平面的速度和IMU与相机之间的时间戳延迟信息）
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        //寻找feature列表中是否有特征点ID对应的特征点
        int feature_id = id_pts.first;
        //find_if函数返回一个迭代器，当查找成功时，该迭代器指向的是第一个符合查找规则的元素；否则该迭代器的指向和第二个参数(即feature.end())的指向相同
        //使用了lambda表达式定义了查找规则
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 表明这是一个新的特征点，将该特征点加入feature列表
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // 表明这是一个老的特征点
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);  
            last_track_num++; // 当前图像帧被跟踪到的特征点数目加1
        }
    }

    // STEP 2 检查滑窗中次新帧是否是关键帧 
    // case1: 若滑动窗口内图像帧的数目小于2或当前图像帧被跟踪到的特征点数目小于20，是关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 计算每个特征在滑窗中次新帧和次次新帧中的视差
    for (auto &it_per_id : feature)
    {
        // 计算的实际上是frame_count-1,也就是前一帧是否为关键帧
        // 因此起始帧至少得是frame_count - 2,同时至少覆盖到frame_count - 1帧
        // 这里不要求该特征点被frame帧检测到
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // case2: 所有特征点被跟踪次数都小于2（即不满足上述计算视差的if条件）的是关键帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        // case3: 所有特征点的平均视差超过设定阈值的为关键帧
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

// 返回frame_count_l与frame_count_r两帧之间的对应特征点
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        // 保证需要的特征点被这两帧都观测到
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            // 获取在feature_per_frame中的索引
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

/**
 * @brief 设置滑窗中所有特征点的深度
 * 
 * @param[in] x 所有特征点的逆深度
 */
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

/**
 * @brief 把给定的逆深度赋值给各个特征点作为深度
 * 
 * @param[in] x 
 */
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

/**
 * @brief 得到所有特征点的逆深度
 * 
 * @return VectorXd 
 */
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

/**
 * @brief 对于所有特征点，利用观测到该特征点的所有图像帧来三角化特征点得到该特征点在第一次被观测到时相机坐标系下的深度
 * 
 * @param[in] Ps   滑动窗口中关键帧的位置 
 * @param[in] tic  平移外参
 * @param[in] ric  旋转外参
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历每一个特征点
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 特征点被跟踪到的次数需大于2并且第一次被观测到时的图像帧在滑窗中的编号小于8
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)  // 代表已经三角化过了
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        // Twi -> Twc,第一个观察到这个特征点的KF的位姿
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];  // t_w_ci = R_w_bi * t_b_c + t_w_bi(可以通过变换矩阵推导)
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];   //R_w_ci = R_w_bi * R_b_c
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        //遍历检测到这个特征点的所有图像帧（这里三角化用的是多帧的计算，而不是两帧）
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++; //第一次 = imu_i = start_frame，imu_i代表起始帧id，imu_j代表当前帧id

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0]; // t_w_ci = R_w_bi * t_b_c + t_w_bi(可以通过变换矩阵推导)
            //R_w_cj = R_w_b * R_b_cj
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0]; //R1 t1为第j帧相机坐标系到世界坐标系的变换矩阵

            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            //R_ci_cj = R_ci_w * R_w_cj
            Eigen::Matrix3d R = R0.transpose() * R1; //R t为第j帧相机坐标系到第i帧相机坐标系的变换矩阵
            Eigen::Matrix<double, 3, 4> P;
            //P为i到j的变换矩阵，即起始帧到当前帧的变换
            //P为T_cj_ci
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            //P = [P0 P1 P2]^T 
            //AX=0      A = [A(2*i) A(2*i+1) A(2*i+2) A(2*i+3) ...]^T
            //A(2*i)   = x(i) * P2 - z(i) * P0
            //A(2*i+1) = y(i) * P2 - z(i) * P1
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            // 没啥用
            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        //对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        // 得到的深度值实际上就是第一个观察到这个特征点的相机坐标系下的深度值
        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH; // 距离太近就设置成默认值（5.0）
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 边缘化次新帧时，对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        // 如果特征被最新帧看到，由于窗口滑动，它的起始帧减1
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            //如果在次新帧之前已经跟踪结束则什么都不做
            if (it->endFrame() < frame_count - 1)
                continue;
            //如果在次新帧仍被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            //如果feature_per_frame为空则直接删除特征点
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 计算某个特征点在次新帧和次次新帧的视差
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // 次次新帧
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // 次新帧

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j; // 归一化相机坐标系的坐标差

    // 又计算了一遍归一化相机坐标系的坐标差
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 计算视差（坐标差的平方）
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}