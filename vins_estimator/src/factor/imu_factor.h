#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>

// 15表示残差的维度，7，9，7，9表示参数块的大小（上一帧的位姿，上一帧的速度和IMU偏置，当前帧位姿，当前帧的速度和IMU偏置）
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
  public: 
    IMUFactor() = delete; // 禁用默认构造函数

    IMUFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
    {
    }

    // 使用ceres解析求导，必须重载这个函数
    // double const *const *parameters表示一个指向常量的指针，该指针又指向一个指向常量的双重指针，其中指针所指向的数据类型为double
    // double **表示一个二级指针（指向指针的指针），用于动态创建和操作二维数组（可以像使用数组一样使用二级指针）
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        // 便于后续计算，把参数块从double数组转换成eigen形式
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]); // 上一帧位置
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); // 上一帧姿态

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);  // 上一帧速度
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]); // 上一帧加速度偏置
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]); // 上一帧陀螺仪偏置

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);  // 当前帧位置
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); // 当前帧姿态

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);  // 当前帧速度
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]); // 当前帧加速度偏置
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]); // 当前帧陀螺仪偏置

//Eigen::Matrix<double, 15, 15> Fd;
//Eigen::Matrix<double, 15, 12> Gd;

//Eigen::Vector3d pPj = Pi + Vi * sum_t - 0.5 * g * sum_t * sum_t + corrected_delta_p;
//Eigen::Quaterniond pQj = Qi * delta_q;
//Eigen::Vector3d pVj = Vi - g * sum_t + corrected_delta_v;
//Eigen::Vector3d pBaj = Bai;
//Eigen::Vector3d pBgj = Bgi;

//Vi + Qi * delta_v - g * sum_dt = Vj;
//Qi * delta_q = Qj;

//delta_p = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi);
//delta_v = Qi.inverse() * (g * sum_dt + Vj - Vi);
//delta_q = Qi.inverse() * Qj;

#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        // Map相当于引用(别名)
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        // 计算IMU测量残差
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                            Pj, Qj, Vj, Baj, Bgj);

        // 将协方差矩阵的逆（协方差）进行cholesky分解，即P^{-1} = L*L^T（其中L是下三角矩阵），sqrt_info保存L^T
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
        //sqrt_info.setIdentity();
        // 实际优化的是r^T*P^{-1}*r,而ceres只接受最小二乘优化，即e^T*e,那么e=L^T*r
        residual = sqrt_info * residual;

        if (jacobians)
        {
            // 获取预积分的误差递推函数中pqv关于ba、bg的Jacobian
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
            }

            // jacobians[0]是residual对parameters[0]（第i帧的IMU位姿 pbi、qbi）的导数
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                // 注意对旋转的雅克比矩阵的维度，虽然待优化参数四元数有四个参数，但是实际时只需要3个参数就能表示旋转
                jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                // IMU预积分只在计算残差时矫正，delta_q并没有变化，所有需要重新矫正
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

                // 雅克比矩阵也要乘上L^T
                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }

            // jacobians[1]是residual对parameters[1]（第i帧的IMU速度和偏置 vbi、bai、bgi）的导数
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

#if 0
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
#else
                //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

                // 雅克比矩阵也要乘上L^T
                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }

            // jacobians[2]是residual对parameters[2]（第j帧的IMU位姿 pbj、qbj）的导数
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

#if 0
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

                // 雅克比矩阵也要乘上L^T
                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }

            // jacobians[3]是residual对parameters[3]（第j帧的IMU速度和偏置 vbj、baj、bgj）的导数
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

                // 雅克比矩阵也要乘上L^T
                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

                //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    IntegrationBase* pre_integration;

};

