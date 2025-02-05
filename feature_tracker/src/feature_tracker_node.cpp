#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_match, pub_restart;
ros::Subscriber sub_img;

FeatureTracker trackerData[NUM_OF_CAM];  //每个相机都有一个FeatureTracker实例，即trackerData[i]
double first_image_time; // 用于控制发布频率的时间戳
int pub_count = 1; //发布的图像帧的数量
bool first_image_flag = true;
double last_image_time = 0;  //上一帧相机的时间戳
bool init_pub = 0;

/**
 * @brief       ROS的回调函数，对新来的图像进行特征点追踪，发布
 * @Description readImage()函数对新来的图像使用光流法进行特征点跟踪,
 *              追踪的特征点封装成feature_points发布到pub_img的话题下，
 *              图像封装成ptr发布在pub_match下
 * @param[in]   img_msg 输入的图像
 * @return      void
*/
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // 对第一帧图像进行特殊处理，用于控制发布频率
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }

    // detect unstable camera stream
    // STEP1 通过连续帧图像的时间戳差判断图像数据流是否稳定，有问题则发布复位消息
    // 图像时间差太多光流追踪就会失败，VINS-Mono没有描述子匹配，因此对时间戳要求更高
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);  // 发布复位消息，通知其他模块重启
        return;
    }

    last_image_time = img_msg->header.stamp.toSec();

    // ROS_INFO("pub_count: %d, timeduration: %f", pub_count, img_msg->header.stamp.toSec() - first_image_time);
    // frequency control
    // STEP2 控制特征点消息发布频率（通过控制间隔时间内的发布次数来控制），并不是每读入一帧图像都要发布特征点
    // !即使不发布也需要进行光流追踪（光流要求图像之间的变化尽可能小）
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)  // round()函数四舍五入，计算实际发布频率（发布数量/间隔时间）
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        // 以相机采集帧率为20Hz为例，如果不重置频率控制变量，上述公式算出的实际频率最终会趋近设置频率，可能会带来误差
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    // 将图像编码从8UC1（八位无符号整型单通道图像）转换为mono8（灰度图），并把ROS message转成cv::Mat
    // 关于8UC1和mono8的区别：8UC1只是说图像有一个大小为8bit的通道（这并不一定是灰度图）;Mono8表示该图像是灰度图,有一个大小为8bit的通道
    // cv_bridge用于把ROS Image message转换成cv::Mat格式,参考http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;                   // 头信息,包含时间戳和坐标系ID
        img.height = img_msg->height;                   // 图像高度，即行数
        img.width = img_msg->width;                     // 图像宽度，即列数
        img.is_bigendian = img_msg->is_bigendian;       // 数据是否是高位编址（将高序字节存储在起始地址），参考https://juejin.cn/post/6930889701507203085
        img.step = img_msg->step;                       // 一行数据的长度（字节）
        img.data = img_msg->data;                       // 实际矩阵数据,大小为step*height
        img.encoding = "mono8";                         // 像素编码（通道含义、排序、大小）
        // toCvCopy()复制数据并返回复制数据地址指针cv_bridge::CvImagePtr
        // toCvShare()获取数据并返回源数据地址指针cv_bridge::CvImageConstPtr,在ROS图像数据编码和参数编码一致时不会发生数据复制
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat show_img = ptr->image; // 浅拷贝,这里image的编码又变成了8UC1（ptr的编码是mono8）

    TicToc t_r; // 计时器

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK) // 单目
            // STEP3 对图像使用光流法进行特征点跟踪以及提取新的特征点
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else // 双目
        {
            if (EQUALIZE)
            {
                // 自适应直方图均衡化处理
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    // STEP4 更新新提取的特征点ID
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)  // 只在单目情形下更新id
                completed |= trackerData[j].updateID(i); // trackerData[j].updateID(i)更新成功返回true,可以直接用=号
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        // STEP5 发布特征点消息，包括时间戳，去畸变归一化坐标，id，像素坐标，归一化平面速度（按一定频率，并不是每帧图像都进行）
        pub_count++;  // 相当于计数器，用于控制频率
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        // 头信息
        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);  // 这个并没有用到

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;  // 当前帧特征点去畸变的归一化坐标
            auto &cur_pts = trackerData[i].cur_pts;  // 当前帧特征点像素坐标
            auto &ids = trackerData[i].ids;  // 当前帧特征点id
            auto &pts_velocity = trackerData[i].pts_velocity;  // 当前帧特征点归一化平面下的速度
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                // 只发布追踪大于1的，因为等于1没法构成重投影约束，也没法三角化
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id); // 这个并没有用到
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    // 利用这个ros消息的格式进行信息存储
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

        // skip the first image; since no optical speed on frist image 
        // 第一帧特征点消息不发布
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);  // 发布特征点消息，包括特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)

        // STEP6 发布用于可视化的特征点图像的消息（按一定频率，并不是每帧图像都进行，越红表示跟踪效果越好，越蓝则越差）
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;  // 图像浅拷贝

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB); // 将灰度图转换为RGB图

                //显示追踪状态，越红越好，越蓝越不行
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());  // 发布特征点图像消息，即在原图像上标出了跟踪到的特征点（越红表示跟踪效果越好，越蓝则越差）
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    // STEP1 ROS初始化
    // 初始化ros，指定节点名称
    ros::init(argc, argv, "feature_tracker");

    /*
    创建节点句柄（节点初始化），并指定这个节点的命名空间（~表示私有命名空间：这个节点的每个话题名称之前都会加上节点名称，
    如/feature_tracker/feature，参考https://blog.csdn.net/lanxiaoke123/article/details/104864379）
    */
    ros::NodeHandle n("~");

    // 设置输出日志的级别，只有级别大于或等于Info的日志消息才会显示输出
    // 输出日志的级别有五种：DEBUG、INFO、WARN、ERROR、FATAL,其中DEBUG和INFO通过stdout输出（可能不会输出到屏幕上），其他通过stderr输出
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    
    // STEP2 读取配置参数
    // 读取yaml文件中关于VINS系统设置的相关参数
    readParameters(n);

    // 读取每个相机的参数，并生成相应的相机实例
    for (int i = 0; i < NUM_OF_CAM; i++) 
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    // 判断是否加入鱼眼mask来去除边缘噪声
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // STEP3 订阅图像topic执行回调函数
    // 订阅话题IMAGE_TOPIC(如/cam0/image_raw),最多缓存100条消息，收到一个message就执行一次回调函数img_callback
    sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    // STEP4 发布特征点相关消息和复位消息
    //在feature话题上发布一个类型为sensor_msgs/PointCloud的消息（跟踪的特征点，用于后端优化），最多缓存1000条消息
    //因为节点初始化时指定了节点的命名空间，因此实际上是在/feature_tracker/feature话题上发布消息，下同
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    //在feature_img话题上发布一个类型为sensor_msgs/Image的消息（跟踪的特征点图像，用于RVIZ显示和调试），最多缓存1000条消息
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    //在restart话题上发布一个类型为std_msgs/Bool的消息（复位信号），最多缓存1000条消息
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);

    // 消息回调函数，与话题订阅函数一同使用
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?