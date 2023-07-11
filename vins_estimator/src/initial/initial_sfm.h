#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SFMFeature
{
    bool state;  //特征点的状态（是否被三角化）
    int id;
    vector<pair<int,Vector2d>> observation;  //所有观测到该特征点的图像帧ID和特征点在这个图像帧的归一化坐标
    double position[3]; //在帧l下的完成三角化的空间坐标
    double depth; //深度
};

// 用于ceres计算重投影误差
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	// const T* const中的第一个const声明了指向常量的指针，此时不能通过指针来改变所指对象的值，但指针本身可以改变，可以指向另外的对象
	// 第二个const声明了指针类型的常量，即指针本身的值不能被改变，不能指向别的对象
	// 函数声明后接的const表示该函数是常成员函数，即该函数不能更新目的对象的数据成员，也不能针对目的对象调用该类中没有用const修饰的成员函数
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const 
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p); // p = R(camera_R) * point，调用该函数时不要钱camera_R是单位四元数，应该会自动归一化
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2]; // 将地图点坐标从世界系转换到相机系
		T xp = p[0] / p[2]; // 归一化
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>( // 2，4，3，3分别表示residuals的维度，camera_R的维度，camera_T的维度和point的维度
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;  // 用于sfm的特征点的数目
};