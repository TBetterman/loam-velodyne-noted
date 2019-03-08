void LaMapping::solveWithGaussNewton(cv::Mat &mat_x,
                                        int iter_count,
                                        pcl::PointCloud<PointType>::Ptr points_selected
                                        pcl::PointCloud<PointType>::Ptr coeff_selected)
{
        int laser_cloud_sel_num = points_selected->size();

        bool is_degenerate = false;
        cv::Mat mat_p(6, 6, CV_32F, cv::Scalar::all(0));

        //预先计算三个欧拉角的sin 和cos, 对应文章中的sin(ex), cos(ex),
        //sin(ey), cos(ey), sin(ez), cos(ez),
        float srx = sin(lidar_pose_in_map_r_[0]);
        float crx = cos(lidar_pose_in_map_r_[0]);
        float sry = sin(lidar_pose_in_map_r_[1]);
        float cry = cos(lidar_pose_in_map_r_[1]);
        float srz = sin(lidar_pose_in_map_r_[2]);
        float crz = cos(lidar_pose_in_map_r_[2]);

        //高斯牛顿求解中用到的一些矩阵， 对应正规方程（normal equation）， AT*A*x = AT*b
        cv::Mat mat_a(laser_cloud_sel_num, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat mat_a_t(6, laser_cloud_sel_num, CV_32F, cv::Scalar::all(0));
        cv::Mat mat_a_t_a(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat mat_b(laser_cloud_sel_num, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat mat_a_t_b(6, 1, CV_32F, cv::Scalar::all(0));

        //将每个观测项构建最小二乘问题, 公式i
        for (int i = 0; i < laser_cloud_sel_num; i++)
        {
                PointType point_ori, coeff;
                point_ori = points_selected->points[i];
                coeff = coeff_selected->points[i];

                //coeff.x, coeff.y, coeff.z 为loss对pose中x, y, z的偏导
                //coeff.intensity 为loss
                //以下代码中的arx计算对应公式h
                float arx = (crx * sry * srz * point_ori.x + crx * crz * sry * point_ori.y - srx * sry * point_ori.z) * coeff.x
                           + (-srx * srz * point_ori.x - crz * srx * point_ori.y - crx * point_ori.z) * coeff.y
                                   + (crx * cry * srz * point_ori.x + crx * cry * crz * point_ori.y - cry * srx * point_ori.z) * coeff.z;
                float ary = ((cry * srx * srz - crz * sry) * point_ori.x + (sry * srz + cry * crz * srx) * point_ori.y + crx * cry * point_ori.z) * coeff.x
                                   + ((-cry * crz - srx * sry * srz) * point_ori.x + (cry * srz - crz * srx * sry) * point_ori.y  - crx * sry * point_ori.z) * coeff.z;
                float arz = ((crz * srx * sry - cry * srz) * point_ori.x  + (-cry * crz - srx * sry * srz) * point_ori.y) * coeff.x
                                   + (crx * crz * point_ori.x - crx * srz * point_ori.y) * coeff.y
                                   + ((sry * srz + cry * crz * srx) * point_ori.x + (crz * sry - cry * srx * srz) * point_ori.y) * coeff.z;

                //见公式i
                mat_a.at<float>(i, 0) = arx;
                mat_a.at<float>(i, 1) = ary;
                mat_a.at<float>(i, 2) = arz;
                mat_a.at<float>(i, 3) = coeff.x;
                mat_a.at<float>(i, 4) = coeff.y;
                mat_a.at<float>(i, 5) = coeff.z;
                mat_b.at<float>(i, 0) = -coeff.intensity;
        }

        //构建normal equation, 见公式k
        cv::transpose(mat_a, mat_a_t);
        mat_a_t_a = mat_a_t * mat_a;
        mat_a_t_b = mat_a_t * mat_b;

        //高斯牛顿法, 直接解normal equation求步长, QR分解是一种解法
        cv::solve(mat_a_t_a, mat_a_t_b, mat_x, cv::DECOMP_QR);

        //具体描述见Loam作者Zhang J的<<On Degeneracy of Optimization-based State Estimation Problems>>
        //大概方法是通过Jacobian的eigenvalue判断哪个分量的约束不足, 不更新那个方向上的迭代
        if (iter_count == 0)
        {
                cv::Mat mat_e(1, 6, CV_32F, cv::Scalar::all(0));
                cv::Mat mat_v(6, 6, CV_32F, cv::Scalar::all(0));
                cv::Mat mat_v2(6, 6, CV_32F, cv::Scalar::all(0));

                cv::eigen(mat_a_t_a, mat_e, mat_v);
                mat_v.copyTo(mat_v2);

                is_degenerate = false;

                float eign_thre[6] = {100, 100, 100, 100, 100, 100};

                for (int i = 5; i >= 0; i--)
                {
                        if (mat_e.at<float>(0, i) < eign_thre[i])
                        {
                                for (int j = 0; j < 6; j++)
                                {
                                        mat_v2.at<float>(i, j) = 0;
                                }

                                is_degenerate = true;
                        }
                        else
                        {
                                break;
                        }
                }

                mat_p = mat_v.inv() * mat_v2;
        }

        if (is_degenerate)
        {
                cv::Mat mat_x2(6, 1, CV_32F, cv::Scalar::all(0));
                mat_x.copyTo(mat_x2);
                mat_x = mat_p * mat_x2;
        }
}

//更新迭代的结果
void LaMapping::updateTransformFromOptimize(cv::Mat &mat_x)
{
        lidar_pose_in_map_r_[0] += mat_x.at<float>(0, 0);
        lidar_pose_in_map_r_[1] += mat_x.at<float>(1, 0);
        lidar_pose_in_map_r_[2] += mat_x.at<float>(2, 0);

        lidar_pose_in_map_t_[0] += mat_x.at<float>(3, 0);
        lidar_pose_in_map_t_[1] += mat_x.at<float>(4, 0);
        lidar_pose_in_map_t_[2] += mat_x.at<float>(5, 0);
}

bool LaMapping::isConverged(cv::Mat &mat_x)
{
        //判断是否已收敛, 这里的判断方法很简单
        float delta_r = sqrt(
                pow(radToDeg(mat_x.at<float>(0, 0)), 2) +
                pow(radToDeg(mat_x.at<float>(1, 0)), 2) +
                pow(radToDeg(mat_x.at<float>(2, 0)), 2));

        float delta_t = sqrt(
                pow(mat_x.at<float>(3, 0) * 100, 2) +
                pow(mat_x.at<float>(4, 0) * 100, 2) +
                pow(mat_x.at<float>(5, 0) * 100, 2));

        return (delta_r < 0.1 && delta_t < 0.3);
}

void LaMapping::doOptimize(int max_iteration)
{
        //复用point cloud结构存储偏导数,
        pcl::PointCloud<PointType>::Ptr coeff_selected boost::make_shared<pcl::PointCloud<PointType>>();
        //存储匹配成功的点,与coeff_selected一一对应
        pcl::PointCloud<PointType>::Ptr points_selected boost::make_shared<pcl::PointCloud<PointType>>();

        //限制迭代次数
        for (int iter_count = 0; iter_count < max_iteration; iter_count++)
        {
                //分别处理corner特征和feature特征, 建立loss成功的点(以下称为有效点), 会加入到features_selected
                //可以见到, 每次迭代, 我们会重新做一次特征点匹配
                procLossAboutCornerPoints(corner_feature_points, points_selected, coeff_selected);
                procLossAboutSurfPoints(surf_feature_points, points_selected, coeff_selected);

                //如果有效点数小于特定值, 我们认为约束不够, 放弃此次优化
                //无法优化的话, 会直接使用pose的预测值
                if (points_selected()->size() < features_slected_num_enough_for_optimize)
                {
                        break;
                }

                cv::Mat mat_x(6, 1, CV_32F, cv::Scalar::all(0));

                //构建norm equation, 求解pose增量
                solveWithGaussNewton(mat_x, iter_count, points_selected, coeff_selected);

                //根据高斯牛顿迭代结果, 更新pose
                updateTransformFromOptimize(mat_x);

                if (isConverged(mat_x))
                {
                        //如果迭代趋于收敛, 退出
                        break;
                }
        }
}

void LaMapping::procLossAboutSurfPoints(pcl::PointCloud<PointType>::Ptr surf_feature_points,
                                        pcl::PointCloud<PointType>::Ptr points_selected,
                                        pcl::PointCloud<PointType>::Ptr coeff_selected
                                        )
{
        for (int i = 0; i < surf_feature_points->size(); i++)
        {
                //将点转换到世界坐标系
                PointType point_sel = transPointToMapCoordinate(surf_feature_points[i]);

                std::vector<int> point_search_ind;
                std::vector<float> point_search_sq_dis;

                //从map对应的kdtree中, 搜索半径一米内的五个surf特征点
                if (surf_kdtree_ptr->radiusSearch(point_sel, 1.0, point_search_ind, point_search_sq_dis, 5) < 5)
                {
                        //没有搜到足够多的点, 本点匹配失败, 此点不对后续的优化贡献约束
                        continue;
                }

                //这些变量用来求解平面方程, 平面方程为AX+BY+CZ+D = 0 <=> AX+BY+CZ=-D <=> (A/D)X+(B/D)Y+(C/D)Z = -1
                //其中(X,Y,Z)是点的坐标, 对应这里的mat_a0, 是已知数
                //A/D, B/D, C/D 对应mat_x0, 是待求的值
                //等式右边的-1对应mat_b0
                cv::Mat mat_a0(5, 3, CV_32F, cv::Scalar::all(0));
                cv::Mat mat_b0(5, 1, CV_32F, cv::Scalar::all(-1));
                cv::Mat mat_x0(3, 1, CV_32F, cv::Scalar::all(0));

                //构建五个最近点的坐标矩阵
                for (int j = 0; j < 5; j++)
                {
                        mat_a0.at<float>(j, 0) = map_ptr->points[point_search_ind[j]].x;
                        mat_a0.at<float>(j, 1) = map_ptr->points[point_search_ind[j]].y;
                        mat_a0.at<float>(j, 2) = map_ptr->points[point_search_ind[j]].z;
                }

                //求解 (A/D)X+(B/D)Y+(C/D)Z = -1 中的 A/D, B/D, C/D
                cv::solve(mat_a0, mat_b0, mat_x0, cv::DECOMP_QR);

                float pa = mat_x0.at<float>(0, 0);
                float pb = mat_x0.at<float>(1, 0);
                float pc = mat_x0.at<float>(2, 0);

                //对应之前的-1, (A/D)X+(B/D)Y+(C/D)Z = -1 <=> (A/D)X+(B/D)Y+(C/D)Z +1 = 0
                float pd = 1;

                //ps为平面法向量的模
                //求得(pa, pb, pc)为法向量, 模为1, pd为平面到原点的距离
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                //确定拟合出的平面与用来拟合的点都足够接近, 表示平面拟合的有效性
                bool plane_valid = false;
                for (int j = 0; j < 5; j++)
                {
                        if (fabs(pa * map_ptr->points[point_search_ind[j]].x +
                                         pb * map_ptr->points[point_search_ind[j]].y +
                                         pc * map_ptr->points[point_search_ind[j]].z + pd) > 0.2)
                        {
                                plane_valid = true;
                                break;
                        }
                }

                if(plane_valid)
                {
                        //平面无效, 本点匹配失败, 此点不对后续的优化贡献约束
                        continue;
                }

                //点到平面的距离, 参考点到平面距离公式, 分母部分为1
                float pd2 = pa * point_sel.x + pb * point_sel.y + pc * point_sel.z + pd;

                //这里的s是个权重, 表示s在这个least-square问题中的置信度, 每个点的置信度不一样
                //理论上, 这个权重, 与点到面距离负相关, 距离越大, 置信度越低, 这里相当于是一个在loss之外加了一个鲁棒性函数, 用来过减弱离群值的影响
                //源代码中"sqrt(sqrt(point_sel.x * point_sel.x + point_sel.y * point_sel.y + point_sel.z * point_sel.z)" 这部分, 并没有什么逻辑性可言
                //你可以设计自己的鲁棒性函数来替代这一行代码
                float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(point_sel.x * point_sel.x + point_sel.y * point_sel.y + point_sel.z * point_sel.z));

                //最终确定的可优化点
                if (s > 0.1)
                {
                        points_selected.push_back(point_sel);

                        //复用PointType传递偏导数, (pa, pb, pc)是法向量,点到平面的垂线, 也是点面距离相对与点坐标的偏导, 详见文章的公式推导部分.
                        //pd2是loss
                        PointType coeff;
                        coeff.x = s * pa;
                        coeff.y = s * pb;
                        coeff.z = s * pc;
                        coeff.intensity = s * pd2;
                        coeff_selected.push_back(corner_feature_points[i]);
                }
                else
                {
                        //距离无效, 本点匹配失败, 此点不对后续的优化贡献约束
                }
        }
}

void LaMapping::procLossAboutCornerPoints(pcl::PointCloud<PointType>::Ptr corner_feature_points,
                                          pcl::PointCloud<PointType>::Ptr points_selected,
                                         pcl::PointCloud<PointType>::Ptr coeff_selected
                                         )
{
        for (int i = 0; i < corner_feature_points->size(); i++)
        {
                //将点转换到世界坐标系
                PointType point_sel = transPointToMapCoordinate(corner_feature_points[i]);


                std::vector<int> point_search_ind;
                std::vector<float> point_search_sq_dis;

                //从map对应的kdtree中, 搜索半径一米内的五个corner特征点
                if (surf_kdtree_ptr->radiusSearch(point_sel, 1.0, point_search_ind, point_search_sq_dis, 5) < 5)
                {
                        //没有搜到足够多的点, 本点匹配失败, 此点不对后续的优化贡献约束
                        continue;
                }

                //将五个最近点的坐标加和求平均
                float cx = 0;
                float cy = 0;
                float cz = 0;

                cv::Mat mat_a1(3, 3, CV_32F, cv::Scalar::all(0));
                cv::Mat mat_d1(1, 3, CV_32F, cv::Scalar::all(0));
                cv::Mat mat_v1(3, 3, CV_32F, cv::Scalar::all(0));

                for (int j = 0; j < 5; j++)
                {
                        cx += corner_map->points[point_search_ind[j]].x;
                        cy += corner_map->points[point_search_ind[j]].y;
                        cz += corner_map->points[point_search_ind[j]].z;
                }

                //坐标均值
                cx /= 5;
                cy /= 5;
                cz /= 5;

                //求均方差
                float a11 = 0;
                float a12 = 0;
                float a13 = 0;
                float a22 = 0;
                float a23 = 0;
                float a33 = 0;

                for (int j = 0; j < 5; j++)
                {
                        float ax = corner_map->points[point_search_ind[j]].x - cx;
                        float ay = corner_map->points[point_search_ind[j]].y - cy;
                        float az = corner_map->points[point_search_ind[j]].z - cz;

                        a11 += ax * ax;
                        a12 += ax * ay;
                        a13 += ax * az;
                        a22 += ay * ay;
                        a23 += ay * az;
                        a33 += az * az;
                }

                //协方差矩阵的6个元素(3*3矩阵, 对角元素重复)
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                //构建协方差矩阵
                mat_a1.at<float>(0, 0) = a11;
                mat_a1.at<float>(0, 1) = a12;
                mat_a1.at<float>(0, 2) = a13;
                mat_a1.at<float>(1, 0) = a12;
                mat_a1.at<float>(1, 1) = a22;
                mat_a1.at<float>(1, 2) = a23;
                mat_a1.at<float>(2, 0) = a13;
                mat_a1.at<float>(2, 1) = a23;
                mat_a1.at<float>(2, 2) = a33;

                //对协方差矩阵进行Eigenvalue decomposition, 以分析空间点的分布规律
                cv::eigen(mat_a1, mat_d1, mat_v1);

                //如果最大特征值相对第二大特征值的比例足够大, 那么反应点的分布趋于一条直线
                if (mat_d1.at<float>(0, 0) > 3 * mat_d1.at<float>(0, 1))
                {
                        //(x0, y0, z0)世界坐标系下的特征点
                        float x0 = point_sel.x;
                        float y0 = point_sel.y;
                        float z0 = point_sel.z;

                        //(x1,y1,z1), (x2,y2,z2) 用来表示特征值最大方向对应的直线
                        float x1 = cx + 0.1 * mat_v1.at<float>(0, 0);
                        float y1 = cy + 0.1 * mat_v1.at<float>(0, 1);
                        float z1 = cz + 0.1 * mat_v1.at<float>(0, 2);
                        float x2 = cx - 0.1 * mat_v1.at<float>(0, 0);
                        float y2 = cy - 0.1 * mat_v1.at<float>(0, 1);
                        float z2 = cz - 0.1 * mat_v1.at<float>(0, 2);

                        //a012为点到直线距离计算分子部分
                        //两向量的叉乘
                        float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
                                         * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
                                         + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
                                         * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
                                         + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))
                                         * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

                        //l12为点到直线距离公式分母部分，(x1,y1,z1), (x2,y2,z2)两点的距离
                        float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

                        //(la, lb, lc)为单位向量, 模为1 ,方向为从垂足->点p
                        float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;
                        float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
                        float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                        //ld2为点到直线的距离
                        float ld2 = a012 / l12;

                        //这里的s是个权重, 表示s在这个least-square问题中的置信度, 每个点的置信度不一样
                        //理论上, 这个权重, 与点到面距离负相关, 距离越大, 置信度越低, 这里相当于是一个在loss之外加了一个鲁棒性函数, 用来过减弱离群值的影响
                        //源代码中只是简单的用1-距离表示权重
                        //你可以设计自己的鲁棒性函数来替代这一行代码
                        float s = 1 - 0.9 * fabs(ld2);

                        if (s > 0.1)
                        {
                                //复用PointType传递偏导数, (la, lb, lc)是点到直线的垂线对应的向量, 也是点线距离相对与点坐标的偏导, 详见文章的公式推导部分.
                                //ld2是loss
                                PointType coeff;
                                coeff.x = s * la;
                                coeff.y = s * lb;
                                coeff.z = s * lc;
                                coeff.intensity = s * ld2;

                                points_selected.push_back(corner_feature_points[i]);
                        }
                        else
                        {
                                //距离无效, 本点匹配失败, 此点不对后续的优化贡献约束
                        }
                }
        }
}

void LaMapping::newLaserProc(pcl::PointCloud<PointType>::Ptr laser)
{
        frame_count_++;

        //第一帧, 完成初始化工作
        if(frame_count_ == 1)
        {
                //记录IMU odom(也可以用其他形式的odometry)
                prepareForNextFrame();

                //第一帧根据当前pose转换到map坐标系下, 作为初始地图
                updateFeatureMaps();

                //地图生成kdtree, 便于后续查找
                updateKdTrees();

                return;
        }

        //每隔几帧处理一次
        if (frame_count_ % 5 == 0)
        {
                //根据odometry进行位姿的predict
                predictTransformByOdom();

                //迭代优化位姿
                doOptimize();

                //迭代结束更新相关的转移矩阵
                prepareForNextFrame();

                //局部地图都要更新
                updateFeatureMaps();

                //地图生成kdtree, 便于后续查找
                updateKdTrees();
        }
}
