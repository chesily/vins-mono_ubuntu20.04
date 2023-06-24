#include "parameters.h"

// 在parameters.h中使用extern关键字声明的变量的定义性声明（定义性声明是唯一的，定义包括分配内存和初始化）
double INIT_DEPTH;    // 深度初始值
double MIN_PARALLAX;  // 关键帧选择阈值（像素单位，根据视差确定）
double ACC_N, ACC_W;  // 加速度计测量的噪声标准差和偏置随机游走标准差
double GYR_N, GYR_W;  // 陀螺仪测量的噪声标准差和偏置随机游走标准差

std::vector<Eigen::Matrix3d> RIC;  //从相机到IMU的旋转矩阵
std::vector<Eigen::Vector3d> TIC;  //从相机到IMU的平移向量

Eigen::Vector3d G{0.0, 0.0, 9.8};  //重力向量

double      BIAS_ACC_THRESHOLD;    //没有用到
double      BIAS_GYR_THRESHOLD;    //没有用到
double      SOLVER_TIME;           //最大解算迭代时间（单位：ms,以保证实时性）
int         NUM_ITERATIONS;        //最大解算迭代次数（以保证实时性）
int         ESTIMATE_EXTRINSIC;    //IMU和相机的外参Rt:0准确；1不准确；2没有
int         ESTIMATE_TD;           //是否在线校准IMU和camera时间
int         ROLLING_SHUTTER;       //1：卷帘快门相机；0：全局快门相机
std::string EX_CALIB_RESULT_PATH;  //相机与IMU外参的输出路径OUTPUT_PATH + "/extrinsic_parameter.csv"
std::string VINS_RESULT_PATH;      //输出路径OUTPUT_PATH + "/vins_result_no_loop.csv"
std::string IMU_TOPIC;             //IMU话题名称，如“/imu0”
double      ROW, COL;              //图片的高和宽
double      TD;                    //IMU和cam的时间差（单位：秒）
double      TR;                    //卷帘快门每帧时间（单位：秒）

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

    // 读取文件中参数的两种方式: >> 或 =
    fsSettings["imu_topic"] >> IMU_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;     // 通过虚拟相机焦距统一标准

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

    // 打开VINS_RESULT_PATH用于输出，但没有操作就关闭了，似乎就是没有用？
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    //IMU和图像相关参数
    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    // 如果完全不知道外参，旋转矩阵初始化为单位矩阵，平移向量初始化为0
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    }
    else 
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);    //将opencv矩阵转换为eigen矩阵
        cv::cv2eigen(cv_T, eigen_T);    //将opencv向量转换为eigen向量
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    // 传感器时间延迟
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    
    fsSettings.release();
}
