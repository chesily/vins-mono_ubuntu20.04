#include "initial_alignment.h"

/**
 * @brief   陀螺仪偏置校正
 * @optional    根据视觉SFM的结果来校正陀螺仪Bias -> Paper V-B-1
 *              主要是将相邻帧之间SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐
 *              注意得到了新的Bias后对应的预积分需要repropagate
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿、预积分量和关于角点的信息
 * @param[out]  Bgs 陀螺仪偏置
 * @return      void
*/
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        //R_ij = (R^c0_bk)^-1 * (R^c0_bk+1) 转换为四元数 q_ij = (q^c0_bk)^-1 * (q^c0_bk+1)
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        //tmp_A = J^R_bw
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        //tmp_b = 2 * (r^bk_bk+1)^-1 * (q^c0_bk)^-1 * (q^c0_bk+1)
        //      = 2 * (r^bk_bk+1)^-1 * q_ij
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        //tmp_A * delta_bg = tmp_b
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }

    //使用eigen库的LDL^T方法（适用于A为正定的对称矩阵的情形）求解陀螺仪零偏Ax=b
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());
    // 滑窗中图像帧的零偏加上求解出来的零偏变化量
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    // 对所有图像帧根据当前零偏重新预积分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

//在半径为G的半球找到切面的一对正交基 
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief   重力矢量细化
 * @optional    重力细化，在其切线空间上用两个变量重新参数化重力 -> Paper V-B-3 
                g^ = ||g|| * (g^-) + w1b1 + w2b2 
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、二自由度重力参数w[w1,w2]^T、尺度s
 * @return      void
*/
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    //限制重力向量的模长
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)  // 迭代四次
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            // 与Ax=b的具体形式也变了
            // tmp_A(6,9) = [-I*dt           0             (R^bk_c0)*dt*dt*b/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ] 
            //              [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt*b                  0                    ]
            // tmp_b(6,1) = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c - (R^bk_c0)*dt*dt*||g||*(g^-)/2 , (b^bk_bk+1)-(R^bk_c0)dt*||g||*(g^-)]^T
            // tmp_A * x = tmp_b 求解最小二乘问题
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

/**
 * @brief   计算尺度，重力加速度和速度
 * @optional    速度、重力向量和尺度初始化Paper -> V-B-2
 *              相邻帧之间的位置和速度与IMU预积分出来的位置和速度对齐，求解最小二乘   
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、重力g、尺度s
 * @return      void
*/
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    //待优化量x的总维度(窗口中每帧的速度V[0:n]、重力g、尺度s)
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        // tmp_A(6,10) = H^bk_bk+1 = [-I*dt           0             (R^bk_c0)*dt*dt/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ] 
        //                           [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt                  0                    ]
        // tmp_b(6,1 ) = z^bk_bk+1 = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c , (b^bk_bk+1)]^T
        // tmp_A * x = tmp_b 求解最小二乘问题
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        //把关于尺度求导得到的雅克比除以100，这就意味着，尺度这个变量对残差的影响力减弱了100倍
        //最终为了能消去残差，优化后的尺度会比实际的大100倍。得到后，要再除以100。这么做的目的，应该是要让尺度的精度更高
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; //H^T*H
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; //H^T*b

        // 与速度状态相关的项
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        // 与重力向量和尺度相关的项
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        // 与速度状态被重力向量和尺度影响的项
        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        // 与重力向量和尺度被速度状态影响的项
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 增强数值稳定性
    A = A * 1000.0;
    b = b * 1000.0;
    //使用eigen库的LDL^T方法（适用于A为正定的对称矩阵的情形）求解Ax=b
    x = A.ldlt().solve(b);

    /* 这里为什么要除以100，就是因为前面tmp_A.block<3, 1>(0, 9)构建的时候除以了100，
    因此最小二乘解出的尺度系数是原来的100倍 */
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);

    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    // 检查优化出的重力大小与9.8之差的绝对值是否大于1，以及尺度是否为负值
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    // 细化重力
    RefineGravity(all_image_frame, g, x);

    // 得到真实尺度
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

/**
 * @brief 
 * 
 * @param[in] all_image_frame 所有图像帧（时间戳到图像帧的映射）
 * @param[out] Bgs 陀螺仪零偏
 * @param[out] g 重力向量
 * @param[out] x 待优化变量，窗口中每帧的速度V[0:n]、重力g、尺度s
 * @return true 
 * @return false 
 */
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    //计算陀螺仪偏置
    solveGyroscopeBias(all_image_frame, Bgs);

    //计算尺度，重力加速度和速度
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
