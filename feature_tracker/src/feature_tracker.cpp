#include "feature_tracker.h"

//将FeatureTracker的static成员变量n_id初始化为0
int FeatureTracker::n_id = 0;

//判断跟踪的特征点是否在图像边界内（比实际图像边界更小）
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    //cvRound()返回跟参数最接近的整数值，即四舍五入
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

//剔除状态位status为0的特征点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//剔除状态位status为0的特征点
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//空的构造函数
FeatureTracker::FeatureTracker()
{
}

/**
 * @brief   对当前帧跟踪到的特征点按照跟踪次数进行排序，并设置掩码为新提取特征点的均匀化做准备            
 * @return  void
*/
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    // 构造（跟踪次数，当前帧特征点坐标，特征点ID）序列
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 排序，跟踪次数越多（越稳定）的点排在越前面（降序排列）(使用了lambda表达式)
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255) // 这个条件好像一定会满足
        {
            // 对位于图像边界内的特征点按照跟踪次数大小重新存储
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // 将特征点周围半径为MIN-DIST（默认为30）的圆内的灰度置0，用于特征点均匀化
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 添加新检测到的特征点n_pts到forw_pts、ids和track_cnt序列中
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1); // 新提取的特征点ID初始化为-1
        track_cnt.push_back(1); // 新提取的特征点被跟踪到的次数初始化为1
    }
}

/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @param[in]   _img 输入图像
 * @param[in]   _cur_time 输入图像的时间戳
 * @return      void
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    //STEP 1 直方图均衡化
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));  //cv::createCLAHE用于生成自适应均衡化的图像
        TicToc t_c;
        clahe->apply(_img, img); // 均衡化操作
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc()); // 记录均衡化花费的时间
    }
    else
        img = _img;

    //如果当前帧的图像数据forw_img为空，说明是第一次读入图像数据
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清空
    forw_pts.clear();

    // STEP 2 LK光流跟踪
    if (cur_pts.size() > 0) 
    {
        TicToc t_o;
        vector<uchar> status; // 特征点的跟踪状态好坏标志位
        vector<float> err;
        // 使用具有金字塔的迭代Lucas-Kanade方法计算稀疏光流，常与角点检测函数cv::goodFeaturesToTrack一同使用
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3); 

        // 剔除跟踪失败的特征点
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i])) // 通过检查是否在图像边界内来剔除外点
                status[i] = 0;
        reduceVector(prev_pts, status);  // 没用到
        reduceVector(cur_pts, status);  // 上一帧图像提取的特征点
        reduceVector(forw_pts, status); // 当前帧图像提取的特征点
        reduceVector(ids, status); // 特征点的ID
        reduceVector(cur_un_pts, status); // 去畸变后的坐标
        reduceVector(track_cnt, status); // 特征点被跟踪到的次数
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc()); // 记录光流跟踪花费的时间
    }

    //被跟踪到的特征在上一帧也存在，追踪数+1(数值越大，说明被跟踪的就越久)
    for (auto &n : track_cnt)
        n++;

    // 如果需要发布该帧的特征点消息，还需进行进一步处理
    if (PUB_THIS_FRAME)
    {
        // STEP 3 通过基础矩阵来剔除外点(可选)
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        // STEP 4 提取新的特征点（可选）
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // 计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // 用于检测shi-tomasi角点（利用上述掩码及MIN_DIST使新提取的特征点分布均匀），常与光流跟踪函数cv::calcOpticalFlowPyrLK一同使用
            /*
             *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值 
             */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    // 没有用
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;

    // 使用当前帧数据更新上一帧数据
    cur_img = forw_img;
    cur_pts = forw_pts;

    // STEP 5 计算当前帧特征点去畸变后的归一化坐标和特征点速度
    undistortedPoints();
    // 使用当前帧时间更新上一帧时间
    prev_time = cur_time;
}

/**
 * @brief   通过对极约束剔除外点 
 * @return  void
*/
void FeatureTracker::rejectWithF()
{
    // 上一帧与当前帧之间至少有八对被光流成功跟踪的特征点才能计算基础矩阵
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            //根据不同的相机模型将二维像素坐标转换到归一化平面无畸变三维坐标
            //也可使用cv::unidistortPoints()函数实现，但效率似乎比该方法低
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            //投影到虚拟相机的像素坐标系（固定焦距），使得F_THRESHOLD与相机无关
            //参考https://github.com/HKUST-Aerial-Robotics/VINS-Mono/issues/48
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算基础矩阵，用于对不满足对极约束的外点进行剔除
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc()); // 记录通过基础矩阵计算剔除外点所花的时间
    }
}

// 更新新提取到的特征点ID
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

// 读取相机参数，生成相机实例
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

// 计算当前帧特征点去畸变后的归一化坐标和特征点速度
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    // 这里处理的仍然是当前帧数据，不过上一帧数据已经被当前帧数据更新了，所以看起来是在处理上一帧数据
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        //根据不同的相机模型将二维像素坐标转换到归一化平面无畸变三维坐标
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    // 计算每个特征点的速度pts_velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            // 如果当前特征点不是新检测到的特征点
            if (ids[i] != -1) 
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    // 得到在归一化平面上的速度（并不是像素平面）
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0)); // 新检测到的特征点速度为0
            }
        }
    }
    else
    {
        // 第一帧的情形
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }

    // 使用当前帧数据更新上一帧数据
    prev_un_pts_map = cur_un_pts_map;
}
