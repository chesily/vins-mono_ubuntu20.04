#pragma once  // 保证该文件只被编译一次，用来防止头文件被重复引用（也可使用#ifndef的方式）
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

// 使用extern关键字声明下面变量能被其他文件使用（引用性声明）
// 头文件中不能定义非static及非const变量，参考https://blog.csdn.net/qq_38988226/article/details/109900487
extern int ROW; //图像高度
extern int COL; //图像宽度
extern int FOCAL_LENGTH; //焦距
const int NUM_OF_CAM = 1; //相机的个数（const类型的变量和类一样，属于内部链接）
extern std::string IMAGE_TOPIC; //图像的ROS TOPIC
extern std::string IMU_TOPIC; //IMU的ROS TOPIC
extern std::string FISHEYE_MASK; //鱼眼相机mask图的位置
extern std::vector<std::string> CAM_NAMES; //相机参数配置文件路径
extern int MAX_CNT; //最大特征点数目
extern int MIN_DIST; //特征点之间的最小间隔
extern int WINDOW_SIZE;
extern int FREQ; //发布跟踪结果的频率
 
extern double F_THRESHOLD; //ransac阈值（像素）
extern int SHOW_TRACK; //是否发布跟踪点图像消息
extern int STEREO_TRACK; //双目跟踪设为1
extern int EQUALIZE; //是否进行直方图均衡化（应对太亮或太暗的场景）
extern int FISHEYE; //使用鱼眼相机设为1
extern bool PUB_THIS_FRAME; //是否发布特征点消息
                         
void readParameters(ros::NodeHandle &n);
