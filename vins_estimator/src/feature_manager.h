#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

/**
* @class FeaturePerFrame
* @brief 特征点类（包括对应于某一图像帧的归一化坐标、像素坐标、在归一化平面的速度和IMU与相机之间的时间戳延迟信息）
*/
class FeaturePerFrame
{
  public:
    //_point:[x,y,z,u,v,vx,vy]
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double   cur_td;        // IMU与相机之间的时间戳延迟
    Vector3d point;         // 特征点的归一化坐标
    Vector2d uv;            // 特征点的像素坐标
    Vector2d velocity;      // 特征点在归一化平面的速度
    double   z;             // 未使用
    bool     is_used;       // 未使用
    double   parallax;      // 未使用
    MatrixXd A;             // 未使用
    VectorXd b;             // 未使用
    double   dep_gradient;  // 未使用
};

/**
* @class 特征点类（以ID为索引）
* @brief ID为feature_id的特征点的所有FeaturePerFrame
*/
class FeaturePerId
{
  public:
    const int feature_id; // 该特征点的ID
    int start_frame;      // 该特征点第一次被检测到时的图像帧在滑窗中的ID
    vector<FeaturePerFrame> feature_per_frame; // 检测到该特征点的图像帧（保存了对应于该特征点的信息）

    int used_num; // 该特征点被跟踪到的次数
    bool is_outlier;
    bool is_margin;
    double estimated_depth; // 该特征点的深度
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)  
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;  // 滑窗中的特征点列表（以ID为索引）
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;  
    Matrix3d ric[NUM_OF_CAM];
};

#endif