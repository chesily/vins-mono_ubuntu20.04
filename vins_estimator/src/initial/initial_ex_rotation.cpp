#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

// imu和相机的旋转外参标定
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    Rc.push_back  (solveRelativeR(corres));             //基于对极几何确定的帧间cam的R
    Rimu.push_back(delta_q_imu.toRotationMatrix());     //基于IMU预积分确定的帧间IMU的R
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);  //基于IMU预积分确定的帧间cam的R

    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG("%d %f", i, angular_distance);

        // 一个简单的核函数
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);  // 作用在残差上面
    }

    //svd分解中最小奇异值对应的右奇异向量（即V的最后一列）作为旋转四元数
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    // 至少迭代了十次且矩阵A第二小的奇异值需大于0.25（需保证矩阵A零空间的秩为1）
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

// 根据两帧特征点求解两帧的旋转矩阵
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    // 特征点匹配对必须大于9（理论上8对匹配点就够了）
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }

        // 求解两帧的本质矩阵（因为这里函数的输入是归一化坐标，想求解基础矩阵输入应该是像素坐标）
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        // SVD分解获得旋转和平移（平移其实没有用上）
        decomposeE(E, R1, R2, t1, t2);

        // 旋转矩阵的行列式应该是1，这里如果是-1就取一下反
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        //通过三角化得到的正深度选择R的解（单目尺度未知，确定t无意义）
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;  // R21（将第一帧的坐标转换为第二帧坐标的旋转矩阵）

        //对R求转置
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen; // 实际想要的是R12（将第二帧的坐标转换为第一帧坐标的旋转矩阵）
    }
    return Matrix3d::Identity();
}

/**
 * @brief 通过三角化来检查R t是否合理
 * 
 * @param[in] l l相机的观测
 * @param[in] r r相机的观测
 * @param[in] R 两个相机之间的旋转
 * @param[in] t 两个相机之间的平移
 * @return double 
 */
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    // 其中一帧设置为单位阵
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    // 第二帧就设置为R t对应的位姿
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));

    // pointcloud为特征点在世界坐标系下的齐次坐标（每一列是一个特征点）
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        // 因为是齐次的，所以要求最后一维等于1
        double normal_factor = pointcloud.col(i).at<float>(3);

        // 得到在各自相机坐标系下的3D坐标
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        // 要求在两个相机下的深度都大于0才是有效特征点
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

// 基于SVD分解本质矩阵得到4组Rt解
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    // 沿z轴旋转-90度得到的旋转矩阵
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    // 沿z轴旋转90度得到的旋转矩阵
    cv::Matx33d Wt(0, 1, 0, 
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
