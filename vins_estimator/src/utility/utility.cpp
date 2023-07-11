#include "utility.h"

// 计算从重力方向到z轴的旋转矩阵（不考虑偏航，偏航角设为0）
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix(); // 从重力方向到z轴的旋转矩阵 ng2 = R0 * ng1（相当于旋转坐标系把世界坐标系旋转到了参考系）
    double yaw = Utility::R2ypr(R0).x(); // 从世界系到参考系的偏航角(度)
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 偏航不可观，补偿回去
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
