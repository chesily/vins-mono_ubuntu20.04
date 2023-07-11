#include "pose_local_parameterization.h"

// 实现了广义加法
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    // Eigen::mapMap用于通过C++中普通的连续指针或者数组(raw C/C++ arrays)来构造Eigen里的Matrix类，这就好比Eigen里的Matrix类的数据和raw C++ array共享了一片地址，也就是引用
    Eigen::Map<const Eigen::Vector3d> _p(x);        // 前三维是平移(x,y,z)
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3); // 后四维是旋转（x,y,z,w）,因为实际上Eigen中四元数的存储顺序为[x y z w]
    // Eigen::Quaterniond初始化的三种方式
    // 第一种方式：Eigen::Quaterniond q1(w, x, y, z);
    // 第二种方式：Eigen::Quaterniond q2(Vector4d(x, y, z, w));
    // 第三种方式：Eigen::Quaterniond q2(Matrix3d(R));


    Eigen::Map<const Eigen::Vector3d> dp(delta); // 平移扰动

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3)); // 旋转扰动

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized(); // 右乘

    return true;
}

// 由于使用解析求导，所以其实不用考虑jacobian矩阵，但LocalParameterization是抽象类，所以需要定义才能实例化
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
