#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr>        imu_buf;      //IMU消息队列
queue<sensor_msgs::PointCloudConstPtr> feature_buf;  //特征点消息队列
queue<sensor_msgs::PointCloudConstPtr> relo_buf;     //回环特征点消息队列
int sum_of_wait = 0;

std::mutex m_buf;       // 数据缓冲区互斥锁
std::mutex m_state;     // 状态互斥锁
std::mutex i_buf;
std::mutex m_estimator; // 状态估计器互斥锁

double latest_time;
Eigen::Vector3d    tmp_P;   // IMU预测的位置
Eigen::Quaterniond tmp_Q;   // IMU预测的姿态
Eigen::Vector3d    tmp_V;   // IMU预测的速度
Eigen::Vector3d    tmp_Ba;  // IMU加速度计偏置
Eigen::Vector3d    tmp_Bg;  // IMU陀螺仪偏置
Eigen::Vector3d    acc_0;   // 上一帧IMU测量加速度
Eigen::Vector3d    gyr_0;   // 上一帧IMU测量角速度
bool   init_feature = 0;
bool   init_imu     = 1;    // 是否是第一帧IMU
double last_imu_t   = 0;

/**
 * @brief 根据当前帧imu数据预测当前位姿和速度
 * @param[in] imu_msg 当前帧IMU消息
 */
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    
    //init_imu=1表示第一个IMU数据
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    // 当前帧IMU测量加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    // 当前帧IMU测量角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    // 上一帧世界坐标系下的加速度
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    // 中值积分角速度
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // 更新当前时刻姿态
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);  

    // 当前帧世界坐标系下的加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    // 中值积分加速度
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 根据加速度和前一时刻的位置速度预测当前时刻的位置和速度
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 使用最新的VIO结果更新IMU状态量，然后以IMU速率重新预测当前帧位姿和速度
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

/**
 * @brief   对imu和图像数据进行对齐并组合
 * @Description     img:    i -------- j  -  -------- k
 *                  imu:    - jjjjjjjj - j/k kkkkkkkk -  
 *                  直到把缓冲区中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据           
 * @return  vector<std::pair<vector<ImuConstPtr>, PointCloudConstPtr>> (IMUs, image)s
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // 对齐标准一：缓冲区中IMU最新数据的时间戳要大于图像特征点最老数据的时间戳
        // 如果IMU还没来
        // imu   ******-
        // image          -****
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // 对齐标准二：缓冲器中IMU最老数据的时间戳要小于图像特征点最老数据的时间戳
        // 如果图像帧没有对应的IMU测量则扔掉这些图像帧
        // imu        -***
        // image    -*****
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        // 此时保证了缓冲区中最老的图像帧（记为当前图像帧）时间戳前后都有IMU测量
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        // 将当前图像帧时间戳之前的IMU数据全部存入IMUs，并将它们从缓冲区删掉
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            // emplace_back和push_back作用相同，但emplace_back能避免临时对象的创建和析构，因此在某些情况下更高效
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        // 等于大于当前图像帧时间戳的第一个IMU测量也存入IMUs,但它不会被删掉，即当前图像帧和下一图像帧共用这个IMU测量
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        // 一帧图像对应多帧IMU测量
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

/**
 * @brief 将最新的IMU消息存进数据缓冲区，同时按照imu频率预测位姿并发送，这样就可以提高里程计频率
 * @param[in] imu_msg 当前帧IMU消息
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //判断IMU时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();       // 上锁
    imu_buf.push(imu_msg);  //将当前帧IMU消息存储IMU消息队列
    m_buf.unlock();     // 解锁 
    con.notify_one();   // 唤醒处于wait中的其中一个条件变量（可能有很多个条件变量，这里只有一个，与存储互斥锁m_buf的细粒度锁lk相关）

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);  //在局部作用域内（此处为{}）自动上锁解锁，防止发生异常时数据死锁（这里构造时给互斥锁m_state上锁，析构时解锁）
        predict(imu_msg); //仅使用IMU测量预测当前帧的位姿和速度
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        // 只有VIO初始化完成后才发送纯IMU预测的位姿和速度（estimator初始化时solver_flag是INITIAL）
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

/**
 * @brief 将前端特征点信息（包括去畸变归一化坐标、ID、像素坐标、归一化平面速度）存进数据缓冲区
 * @param[in] feature_msg 当前帧特征点消息
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    // TODO前两帧图像不接收（第一帧图像没有追踪大于1的特征点所以都没有发布，第二帧为什么不接收呢？）
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();   //上锁
    feature_buf.push(feature_msg);
    m_buf.unlock(); //解锁
    con.notify_one();//唤醒process线程(有新数据了，要继续干活了)
}

/**
 * @brief 将vins估计器复位
 * @param[in] restart_msg 
 */
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        // 清空数据缓冲区的所有IMU和特征点消息
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        // 重置状态估计器
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

//relocalization回调函数，将points_msg放入relo_buf   
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
/**
 * @brief   VIO的主线程
 * @Description 等待并获取measurements：(IMUs, img_msg)s，计算dt
 *              estimator.processIMU()进行IMU预积分         
 *              estimator.setReloFrame()设置重定位帧
 *              estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化     
 * @return      void
*/
void process()
{
    while (true)   // 这个线程会一直循环下去
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        // unique_lock也实现了自动上锁解锁，但比lock_guard更加灵活，能记录现在处于上锁还是没上锁状态，在析构的时候，会根据当前状态来决定是否要进行解锁（局部作用域内可以重复上锁解锁）
        std::unique_lock<std::mutex> lk(m_buf); 
        // wait()函数会先调用互斥锁的unlock()函数，然后再将自己睡眠，在被唤醒后，又会继续持有锁，保护后面的队列操作
        // lock_guard没有lock和unlock接口，而unique_lock提供了，这就是此时必须使用unique_lock的原因
        // 后面接的lambda表达式如果返回true则wait函数会被直接唤醒，该线程才会继续运行
        // 条件变量用于多线程中存取数据，具体可参考https://www.jianshu.com/p/c1dfa1d40f53
        // 提取measure时互斥锁会被锁住，此时缓冲区无法接收IMU和图像特征点数据
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock(); // 解锁，IMU和图像特征点的回调函数此时才能继续接收数据

        m_estimator.lock(); // 状态估计其上锁，进行状态估计操作，防止与复位操作冲突
        // 基于范围的for循环，遍历每对image IMUs组合进行IMU预积分，并处理回环
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            // 遍历与一帧图像对应的IMU数据
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t) // 如果IMU的时间戳小于等于图像的时间戳
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time; //IMU间隔时间
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // IMU预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                // 对于大于图像帧的时间戳的IMU数据（在每对image imus组合中只有一帧或零帧IMU数据是这样的，除非最后一帧IMU时间戳刚好与图像帧时间戳重合才是零帧）
                else 
                {
                    double dt_1 = img_t - current_time; // 图像帧与上一帧IMU之间的时间差
                    double dt_2 = t - img_t; // 当前帧IMU与图像帧之间的时间差
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    // 使用线性插值计算出到图像帧时间戳截止的IMU测量
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    // IMU预积分
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 回环相关部分
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())   // TODO取出缓冲区中最老的回环帧（这里为什么不用上锁）
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id; // 回环帧特征点的去畸变归一化坐标
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                // 回环帧的位姿
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            // 特征点id映射到特征点信息
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5; // TODO特征点的ID+0.5（这里有隐式类型转换，应该向下取整，那是不是加不加0.5都无所谓）
                int feature_id = v / NUM_OF_CAM;    
                int camera_id = v % NUM_OF_CAM;     // 相机ID始终为0
                double x = img_msg->points[i].x;    // 去畸变的归一化坐标
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i]; // 特征点像素坐标
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i]; // 特征点在归一化平面的速度
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);     // 检查是不是归一化
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            // 处理图像特征
            estimator.processImage(image, img_msg->header);

            // 打印相机位姿、外参、VO花费的时间、运行的轨迹长度和时间戳延迟
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);

            // 给RVIZ发送topic
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            pubOdometry  (estimator, header);   
            pubKeyPoses  (estimator, header);   
            pubCameraPose(estimator, header);  
            pubPointCloud(estimator, header);  
            pubTF        (estimator, header);   
            pubKeyframe   (estimator);         
            if (relo_msg != NULL)
                pubRelocalization(estimator);   
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();

        // 更新IMU速率级里程计估计结果
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();  
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    // STEP1 ROS初始化
    // 初始化ros，指定节点名称
    ros::init(argc, argv, "vins_estimator");
    /*
    创建节点句柄（节点初始化），并指定这个节点的命名空间（~表示私有命名空间：这个节点的每个话题名称之前都会加上节点名称，
    如/vins_estimator/odometry，参考https://blog.csdn.net/lanxiaoke123/article/details/104864379）
    */
    ros::NodeHandle n("~");

    // 设置输出日志的级别，只有级别大于或等于Info的日志消息才会显示输出
    // 输出日志的级别有五种：DEBUG、INFO、WARN、ERROR、FATAL,其中DEBUG和INFO通过stdout输出（可能不会输出到屏幕上），其他通过stderr输出
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // STEP2 读取配置参数
    readParameters(n);

    // STEP3 设置估计器参数
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // STEP4 注册待发布的topic
    registerPub(n);

    // STEP5 订阅与IMU、前端处理的特征点、前端重启指令、回环检测的快速重定位相关的topic,并执行相应的回调函数
    // 订阅话题IMU_TOPIC(如/imu0),执行回调函数imu_callback，ros::TransportHints().tcpNoDelay()表示如果使用了TCP传输，那么指定使用TCP_NODELAY来提供潜在的低延迟连接
    // 最多缓存2000条消息，收到一个message就执行一次回调函数，下同
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    // 订阅话题/feature_tracker/feature（即前端跟踪的特征点信息），并执行回调函数feature_callback
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // 订阅话题/feature_tracker/restart（即复位信号），并执行回调函数restart_callback
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // 订阅话题/pose_graph/match_points（即回环检测的fast relocalization响应），并执行回调函数relocalization_callback
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    // STEP6 执行紧耦合的单目VIO
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
