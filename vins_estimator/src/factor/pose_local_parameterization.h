#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

// 公有继承，基类的公有和保护成员的访问属性在派生类中不变，而基类的私有成员不可直接访问
class PoseLocalParameterization : public ceres::LocalParameterization
{
    // 派生类的虚函数会覆盖基类的虚函数，还会隐藏基类中同名函数的所有其他重载形式
    // 派生类覆盖基类的成员函数时，既可以使用virtual关键字，也可以不使用，二者没有差别
    // 很多人习惯于在派生类的函数中也使用virtual关键字，因为这样可以清楚地提示这是一个虚函数
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const; // 函数实现了⊞(x,Δ)
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;  // a GlobalSize() * LocalSize() matrix以行的形式存储（即一个数组指针）
    virtual int GlobalSize() const { return 7; }; // 返回x的尺寸(参数块所在的环境空间的维度)
    virtual int LocalSize() const { return 6; };  // 返回delta的尺寸(切空间的维度)
};

// 额外提醒
// 在ceres中没有明确的说明之处都认为矩阵raw memory存储方式Row Major的，这与Eigen默认的Col Major是相反的
// ceres默认的Quaternion row memory存储方式是w,x,y,z，而Eigen Quaternion的存储方式是x,y,z,w，这就导致在ceres代码中除ceres::QuaternionParameterization之外还有ceres::EigenQuaternionParameterization