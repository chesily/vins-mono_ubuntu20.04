#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;       //相机的个数（const类型的变量和类一样，属于内部链接）
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

// 使用extern关键字声明下面变量能被其他文件使用（引用性声明）
// 头文件中不能定义非static及非const变量，参考https://blog.csdn.net/qq_38988226/article/details/109900487
extern double INIT_DEPTH;       // 深度初始值
extern double MIN_PARALLAX;     // 关键帧选择阈值（像素单位，根据视差确定）
extern int ESTIMATE_EXTRINSIC;  // IMU和相机的外参Rt:0准确；1不准确；2没有

extern double ACC_N, ACC_W;  // 加速度计测量的噪声标准差和偏置随机游走标准差
extern double GYR_N, GYR_W;  // 陀螺仪测量的噪声标准差和偏置随机游走标准差

extern std::vector<Eigen::Matrix3d> RIC;  //从相机到IMU的旋转矩阵
extern std::vector<Eigen::Vector3d> TIC;  //从相机到IMU的平移向量
extern Eigen::Vector3d G;                 //重力向量

extern double BIAS_ACC_THRESHOLD;         //没有用到
extern double BIAS_GYR_THRESHOLD;         //没有用到
extern double SOLVER_TIME;                //最大解算迭代时间（单位：ms,以保证实时性）
extern int NUM_ITERATIONS;                //最大解算迭代次数（以保证实时性）
extern std::string EX_CALIB_RESULT_PATH;  //相机与IMU外参的输出路径OUTPUT_PATH + "/extrinsic_parameter.csv"
extern std::string VINS_RESULT_PATH;      //输出路径OUTPUT_PATH + "/vins_result_no_loop.csv"
extern std::string IMU_TOPIC;             //IMU topic名"/imu0"
extern double TD;                         //IMU和cam的时间差（单位：秒，readed image clock + td = real image clock (IMU clock)）
extern double TR;                         //卷帘快门每帧时间（单位：秒）
extern int ESTIMATE_TD;                   //是否在线校准IMU和camera时间
extern int ROLLING_SHUTTER;               //1：卷帘快门相机；0：全局快门相机
extern double ROW, COL;                   //图像的高和宽


void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
