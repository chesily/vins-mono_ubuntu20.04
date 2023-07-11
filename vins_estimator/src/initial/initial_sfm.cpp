#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

/**
 * @brief 对单一特征点三角化（使用svd计算其世界坐标）
 * 
 * @param[in] Pose0 两帧位姿
 * @param[in] Pose1 
 * @param[in] point0 特征点在两帧下的观测
 * @param[in] point1 
 * @param[out] point_3d 三角化结果
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	// 通过奇异值分解求解Ax=0得到
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	// 齐次向量归一化
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


/**
 * @brief 以上一帧的位姿作为初值通过pnp求解当前帧的位姿
 * 
 * @param[in] R_initial 旋转初值
 * @param[in] P_initial 平移初值
 * @param[in] i 	当前帧的索引
 * @param[in] sfm_f 	所有特征点的信息
 * @return true 
 * @return false 
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++) //要把待求帧i上所有特征点的归一化坐标和3D坐标(l系上)都找出来
	{
		if (sfm_f[j].state != true) // 是false就是没有被三角化，pnp是3d到2d求解，因此需要3d点
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i) //如果该特征点在帧i上出现过
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);  // 将旋转矩阵转换为旋转向量
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);  //得到了世界坐标系到第i帧的旋转平移
	if(!pnp_succ) 
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

/**
 * @brief 根据两帧索引和位姿计算匹配特征点在世界坐标系下的3D坐标
 * 
 * @param[in] frame0 
 * @param[in] Pose0 
 * @param[in] frame1 
 * @param[in] Pose1 
 * @param[in] sfm_f 
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)  //如果这个特征已经三角化过了，那就跳过
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		// 遍历该特征点的观测，看看是不是两帧都能看到
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second; // 该特征点对应与frame0图像帧的归一化坐标
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second; // 该特征点对应与frame1图像帧的归一化坐标
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
/**
 * @brief   根据滑窗中参考帧（第l帧）和最新帧的位姿变换，得到滑窗中所有帧的位姿和特征点的3d点坐标，最后通过ceres进行优化
 * @param[in]   frame_num	窗口总帧数（frame_count + 1）
 * @param[out]  q 	窗口内图像帧的旋转四元数q（相对于第l帧）//q_w_ck
 * @param[out]	T 	窗口内图像帧的平移向量T（相对于第l帧）//T_w_ck
 * @param[in]  	l 	参考帧的索引
 * @param[in]  	relative_R	最新帧到第l帧（参考帧）的旋转矩阵
 * @param[in]  	relative_T 	最新帧到第l帧（参考帧）的平移向量
 * @param[in]  	sfm_f		用于sfm的所有特征点
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return  bool true:sfm求解成功
*/
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	//假设第l帧为原点，根据最新帧到第l帧的relative_R，relative_T，得到最新帧位姿
	//relative_R和relative_T设为T_l_ck
	//q[l]和T[l]设为T_w_l
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	// T_w_ck = T_w_l * T_l_ck
	q[frame_num - 1] = q[l] * Quaterniond(relative_R); //frame_num-1表示最新图像帧（frame_num = frame_count + 1）
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	// 由于纯视觉slam处理都是Tcw,因此下面把Twc转成Tcw
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	//T_l_w
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	//T_ck_w
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	// Step 1： 求解滑窗中参考帧到最新帧之间所有图像帧的位姿并三角化对应特征点
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			//T_ci_w
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) // 以前一帧的位姿作为初值使用pnp求解当前帧的位姿
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		// 通过第i帧图像和最新帧图像的特征匹配关系三角化特征点
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	//3: triangulate l-----l+1 l+2 ... frame_num -2
	// Step 2: 通过第i帧图像和第l帧图像的特征匹配关系三角化特征点（补充地图点）
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l

	// step 3: 求解滑窗中参考帧之前的所有图像帧并三角化对应特征点
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) // 以后一帧的位姿作为初值使用pnp求解当前帧的位姿
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		// 通过第i帧图像和第l帧图像的特征匹配关系三角化特征点
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	// STEP4: 得到了所有关键帧的位姿，遍历没有被三角化的特征点，进行三角化
	// 至此得到了滑动窗口中所有图像帧的位姿以及特征点的3d坐标
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2) // 只有被两个以上的图像帧观测到才可以三角化
		{
			Vector2d point0, point1;
			// 取首尾两个图像帧，尽量保证两图像帧之间足够视差
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	// STEP5: 求出滑窗中所有位姿和3d点之后，进行一个视觉slam的global BA
	ceres::Problem problem;
	// LocalParameterization已经被弃用，在2.2.0版本被移除，可用Maniford替代
	// QuaternionParameterization是LocalParameterization的一个实例，使用四元数参数化旋转（四维空间的三维流形）
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization(); 
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		//加入待优化量：全局位姿（地图点会被隐式地加入从而被优化）
		//T_ck_w
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		/*
		void Problem::AddParameterBlock(double *values, int size, LocalParameterization *local_parameterization)
		已经被弃用，在2.2.0版本被移除，可用基于Manifold的AddParameterBlock版本替代
		为问题添加具有适当大小和参数化的参数块，local_parameterization可以为nullptr
		*/
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		// void Problem::AddParameterBlock(double *values, int size)
		problem.AddParameterBlock(c_translation[i], 3);
		// 由于是单目视觉slam，有七个自由度不可观，因此，fix一些参数块避免在零空间漂移
		// fix设置的世界坐标系第l帧的位姿，同时fix最后一帧的位移用来fix尺度
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	// 只有视觉重投影构成约束，因此遍历所有的特征点，构建约束
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)  // 必须是三角化之后的
			continue;
		// 遍历所有的观测帧，对这些帧建立约束
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());
			// 约束了这一帧位姿和3d地图点（NULL表示没有用鲁棒核函数）
			// 如果没有声明AddParameterBlock，那么默认形参列表中的参数会被隐式地加入参数块，所以地图点也会被优化
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}

	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}

	//这里得到的是第l帧坐标系到各帧的变换矩阵，应将其转变为各帧在第l帧坐标系上的位姿(Tcw->Twc)
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	// 将成功三角化的地图点保存
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

