#include "parameters.h"

// 在parameters.h中使用extern关键字声明的变量的定义性声明（定义性声明是唯一的，定义包括分配内存和初始化）
std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;  //配置文件路径
std::string FISHEYE_MASK;
int    MAX_CNT;         //最大特征点数目
int    MIN_DIST;        //特征点之间的最小间隔
int    WINDOW_SIZE;     // 可视化相关
int    FREQ;            //发布跟踪结果的频率
double F_THRESHOLD;     //ransac阈值（像素）
int    SHOW_TRACK;      //是否发布跟踪点图像消息
int    STEREO_TRACK;    //双目跟踪设为1
int    EQUALIZE;        //是否进行直方图均衡化（应对太亮或太暗的场景）
int    ROW;             //图像高度
int    COL;             //图像宽度
int    FOCAL_LENGTH;    //焦距
int    FISHEYE;         //使用鱼眼相机设为1
bool   PUB_THIS_FRAME;  //是否发布特征点

// 该模板函数用于检索参数对应的参数值
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

// 读取配置参数，通过roslaunch文件的参数服务器获得
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    /* 
    获取配置文件的路径（"config_file"的具体路径在.launch文件中定义，
    以euroc.launch为例，具体路径为$(find feature_tracker)/../config/euroc/euroc_config.yaml）
    */ 
    config_file = readParam<std::string>(n, "config_file");
    // 实例化一个FileStorage对象，打开配置文件进行读操作（从XML或YAML文件中读取数据）
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    // 检查文件是否已经打开
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    // 读取文件中参数的两种方式: >> 或 =
    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"]; // 是否做均衡化处理
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
