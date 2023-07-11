#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

/**
* @class Estimator 状态估计器
* @Description IMU预积分，图像IMU融合的初始化和状态估计，重定位
*/
class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,   // 表明系统还未初始化
        NON_LINEAR // 表明系统已经初始化
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;  // 重力向量
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];   // 从相机到IMU的旋转矩阵数组
    Vector3d tic[NUM_OF_CAM];   // 从相机到IMU的平移向量数组

    Vector3d Ps[(WINDOW_SIZE + 1)];   // 从滑窗在第0帧到第i帧的平移（在参考帧相机系下的表示）
    Vector3d Vs[(WINDOW_SIZE + 1)];   // 滑动窗口中所有关键帧速度（从当前帧imu系到参考帧相机系）
    Matrix3d Rs[(WINDOW_SIZE + 1)];   // 滑动窗口中所有关键帧姿态（从当前帧imu系到参考帧相机系）
    Vector3d Bas[(WINDOW_SIZE + 1)];  // 滑动窗口中所有关键帧对应的加速度计偏置
    Vector3d Bgs[(WINDOW_SIZE + 1)];  // 滑动窗口中所有关键帧对应的陀螺仪偏置
    double td;      // IMU和相机的时间戳延迟（单位：秒,cam_clock + td = imu_clock）(使用参数文件中的参数初始化)

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];  // 滑动窗口中关键帧的图像头信息

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; // 关键帧的预积分量（初始化为空指针）
    Vector3d acc_0, gyr_0;  // 上一帧IMU测量

    //滑窗中的dt,a,v
    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count; // 滑动窗口内图像帧的数量（初始化为0）
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;  // 是否为第一帧IMU（初始化为false）
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;  // 上次执行VIO初始化时的图像时间戳


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];  // 滑窗中的位姿参数块（共11帧）
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];  // 滑窗中的速度、IMU偏置参数块（共11帧）
    double para_Feature[NUM_OF_F][SIZE_FEATURE]; 
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];  // 相机和IMU外参的参数块（共1个相机）
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];  // 相机与IMU的时间戳延迟参数块
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;   // 上次边缘化所有信息
    vector<double *> last_marginalization_parameter_blocks;  // 上次边缘化参数块

    map<double, ImageFrame> all_image_frame; // 所有图像帧对象（时间戳到图像帧的映射）
    IntegrationBase *tmp_pre_integration; // 临时预积分量（初始化为空指针，只保存与最新图像帧相关的预积分量）

    //relocalization variable 重定位所需的变量
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
