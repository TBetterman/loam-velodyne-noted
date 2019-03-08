//lidar坐标系:x轴向前，y轴向左，z轴向上的右手坐标系
//处理频率:与激光帧率一致
void FeatureDt::getFeaturePoints(pcl::PointCloud<PointType> &laser_cloud_in,
                                          uint64_t ts,
                                          pcl::PointCloud<PointType>::Ptr &laser_cloud_in_range,
                                          pcl::PointCloud<PointType>::Ptr &corner_sharp,
                                          pcl::PointCloud<PointType>::Ptr &corner_less_sharp,
                                          pcl::PointCloud<PointType>::Ptr &surf_flat,
                                          pcl::PointCloud<PointType>::Ptr &surf_less_flat)
{
  int cloud_in_size = laser_cloud_in.points.size();

  //获取点云的开始和结束水平角度, 确定一帧中点的角度范围
  //此处需要注意一帧扫描角度不一定<2pi, 可能大于2pi, 角度需特殊处理
  //角度范围用于确定每个点的相对扫描时间, 用于运动补偿
  float start_yaw;
  float end_yaw;
  getYawRange(laser_cloud_in, cloud_in_size, start_yaw, end_yaw);

  //至于此处half_passed的作用, 文中有详述
  bool half_passed = false;

  int cloud_filted_size = cloud_in_size;
  PointType point;

  //每一线存储为一个单独的线(SCAN), 针对单个线计算特征点
  std::vector<PointCloudType> laser_cloud_per_scan(N_SCANS);

  //把点根据几何角度(竖直)分配到线中
  for (int i = 0; i < cloud_in_size; i++)
  {
    point.x = laser_cloud_in.points[i].x;
    point.y = laser_cloud_in.points[i].y;
    point.z = laser_cloud_in.points[i].z;

    //scan_id由竖直角度映射而得, 具体参考lidar的user manual
    int scan_id = getScanIDOfPoint(point);
    if (scan_id > (N_SCANS - 1) || scan_id < 0)
    {
      cloud_filted_size--;
      continue;
    }

    float point_yaw;
    getYawOfOnePoint(start_yaw, end_yaw, point, point_yaw, half_passed);

    //计算此点在此帧中扫描的相对时间, 用来作为后续的运动补偿
    float yaw_percent = (point_yaw - start_yaw) / (end_yaw - start_yaw);

    //复用这个字段传递参数,整数部分是SCANID, 小数部分是相对时间
    point.intensity = toIntensity(scan_id, yaw_percent);

    //往对应的scan中, 新增一个point
    laser_cloud_per_scan[scan_id].push_back(point);
  }

  //将点云按scan顺序排列, 重新组成大点云
  for (int i = 0; i < N_SCANS; i++)
  {
    *laser_cloud_in_range += laser_cloud_per_scan[i];
  }

  //记录每个scanid的开始和结束点
  std::vector<int> scan_start_ind(N_SCANS, 0);
  std::vector<int> scan_end_ind(N_SCANS, 0);

  calcCurvature(laser_cloud_in_range, cloud_filted_size, scan_start_ind, scan_end_ind);

  detectFeaturePoint(laser_cloud_in_range, cloud_filted_size, scan_start_ind, scan_end_ind, corner_sharp, corner_less_sharp, surf_flat, surf_less_flat);
}

//这三个函数涉及服用点云中点的intersity字段,存储scanid(intensity的整数部分)和相对扫描时间(intensity小数部分)
inline float toIntensity(int scanID, float yawPercent)
{
    return scanID + yawPercent;
}

inline int toScanID(float intensity)
{
    return int(intensity);
}

inline float toReltiveTime(float intensity)
{
    return intensity - int(intensity);
}

void FeatureDt::getYawRange(pcl::PointCloud<PointType> &laser_cloud_in,
                            int cloud_size,
                            float &start_yaw, float &end_yaw
                            )
{
  //第一个点和最后一个点对应的是第一和最末一束线
  //velodyne是顺时针增大, 而坐标轴中的yaw是逆时针增加, 所以这里要取负号
  start_yaw = -atan2(laser_cloud_in.points[0].y, laser_cloud_in.points[0].x);
  end_yaw = -atan2(laser_cloud_in.points[cloud_size - 1].y, laser_cloud_in.points[cloud_size - 1].x) + 2 * M_PI;

  //atan2得到的角度是[-pi, pi], 所以， 如果都用标准化的角度， 那么end_yaw可能小于或者接近start_yaw， 这不利于后续的运动补偿
  //因为运动补偿需要从每个点的水平角度确定其在一帧中的相对时间
  //我们需要转换到end_yaw > start_yaw 且end_yaw-start_yaw接近2*M_PI的形式， 所以就有了以下代码
  if (end_yaw - start_yaw > 3 * M_PI)
  {
    end_yaw -= 2 * M_PI;
  }
  else if (end_yaw - start_yaw < M_PI)
  {
    end_yaw += 2 * M_PI;
  }
}

//yaw决定了点的相对扫描时间
void FeatureDt::getYawOfOnePoint(float &start_yaw,
                                  float &end_yaw,
                                  PointType point,
                                  float &yaw,
                                  bool &half_passed
                                   )
{
  yaw = -atan2(point.y, point.x);

  //因为转一圈可能会超过2pi， 故角度a可能对应a或者2pi + a
  //如何确定是a还是2pi+a呢， half_passed 利用点的顺序与时间先后相关这一点解决了这个问题
  if (!half_passed)
  {
    if (yaw < start_yaw - M_PI / 2)
    {
      yaw += 2 * M_PI;
    }
    else if (yaw > start_yaw + M_PI * 3 / 2)
    {
      yaw -= 2 * M_PI;
    }

    if (yaw - start_yaw > M_PI)
    {
      half_passed = true;
    }
  }
  else
  {
    yaw += 2 * M_PI;

    if (yaw < end_yaw - M_PI * 3 / 2)
    {
      yaw += 2 * M_PI;
    }
    else if (yaw > end_yaw + M_PI / 2)
    {
      yaw -= 2 * M_PI;
    }
  }
}

//计算曲率
void FeatureDt::calcCurvature(pcl::PointCloud<PointType>::Ptr laser_cloud,
                             int cloud_size,
                             std::vector<int> &scan_start_ind,
                             std::vector<int> &scan_end_ind
                            )
{
  int scan_count = -1;

  //针对每个点求其特征
  for (int i = 5; i < cloud_size - 5; i++)
  {
    //用周围的10个点计算其描述子, 边界的点省略掉， 因为他们周围的点不足5个
    float diff_x = laser_cloud->points[i - 5].x + laser_cloud->points[i - 4].x + laser_cloud->points[i - 3].x + laser_cloud->points[i - 2].x + laser_cloud->points[i - 1].x - 10 * laser_cloud->points[i].x + laser_cloud->points[i + 1].x + laser_cloud->points[i + 2].x + laser_cloud->points[i + 3].x + laser_cloud->points[i + 4].x + laser_cloud->points[i + 5].x;
    float diff_y = laser_cloud->points[i - 5].y + laser_cloud->points[i - 4].y + laser_cloud->points[i - 3].y + laser_cloud->points[i - 2].y + laser_cloud->points[i - 1].y - 10 * laser_cloud->points[i].y + laser_cloud->points[i + 1].y + laser_cloud->points[i + 2].y + laser_cloud->points[i + 3].y + laser_cloud->points[i + 4].y + laser_cloud->points[i + 5].y;
    float diff_z = laser_cloud->points[i - 5].z + laser_cloud->points[i - 4].z + laser_cloud->points[i - 3].z + laser_cloud->points[i - 2].z + laser_cloud->points[i - 1].z - 10 * laser_cloud->points[i].z + laser_cloud->points[i + 1].z + laser_cloud->points[i + 2].z + laser_cloud->points[i + 3].z + laser_cloud->points[i + 4].z + laser_cloud->points[i + 5].z;

    //曲率计算公式
    cloud_curvature_[i] = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
    cloud_sort_ind_[i] = i;

    //特征点默认为less_smooth, 后续会在曲率筛选中改变这个label
    //neighbor_picked意义在于,当一个点被选为corner或者surf时, 周边N个点不要被选, 让特征点尽量分布广, 不聚集
    cloud_neighbor_picked_[i] = 0;
    cloud_label_[i] = CURV_LESS_SMOOTH;

    if (toScanID(laser_cloud->points[i].intensity) != scan_count)
    {
      scan_count = toScanID(laser_cloud->points[i].intensity);

      if (scan_count > 0 && scan_count < N_SCANS)
      {
        //设置本scan的头
        scan_start_ind[scan_count] = i + 5;

        //设置上个scan的尾
        scan_end_ind[scan_count - 1] = i - 5;
      }
    }
  }

  //最前和最后5个点不方便计算曲率, 抛弃
  //不要认为一个SCAN首尾可以相接, 运动状态导致的畸变会使得首尾差距很大
  scan_start_ind[0] = 5;
  scan_end_ind.back() = cloud_size - 5;

  //paper中(a) (b)这两种特殊情况的点不会被选为corner orsurface
  for (int i = 5; i < cloud_size - 6; i++)
  {
    //计算曲率
    float diff_x = laser_cloud->points[i + 1].x - laser_cloud->points[i].x;
    float diff_y = laser_cloud->points[i + 1].y - laser_cloud->points[i].y;
    float diff_z = laser_cloud->points[i + 1].z - laser_cloud->points[i].z;
    float diff = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

    //曲率阈值过滤
    if (diff > 0.1)
    {
      float depth1 = sqrt(laser_cloud->points[i].x * laser_cloud->points[i].x +
                          laser_cloud->points[i].y * laser_cloud->points[i].y +
                          laser_cloud->points[i].z * laser_cloud->points[i].z);

      float depth2 = sqrt(laser_cloud->points[i + 1].x * laser_cloud->points[i + 1].x +
                          laser_cloud->points[i + 1].y * laser_cloud->points[i + 1].y +
                          laser_cloud->points[i + 1].z * laser_cloud->points[i + 1].z);

      //针对paper中(b)情况
      if (depth1 > depth2)
      {
        diff_x = laser_cloud->points[i + 1].x - laser_cloud->points[i].x * depth2 / depth1;
        diff_y = laser_cloud->points[i + 1].y - laser_cloud->points[i].y * depth2 / depth1;
        diff_z = laser_cloud->points[i + 1].z - laser_cloud->points[i].z * depth2 / depth1;

        if (sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) / depth2 < 0.1)
        {

          cloud_neighbor_picked_[i - 5] = 1;
          cloud_neighbor_picked_[i - 4] = 1;
          cloud_neighbor_picked_[i - 3] = 1;
          cloud_neighbor_picked_[i - 2] = 1;
          cloud_neighbor_picked_[i - 1] = 1;
          cloud_neighbor_picked_[i] = 1;
        }
      }
      else
      {
        diff_x = laser_cloud->points[i + 1].x * depth1 / depth2 - laser_cloud->points[i].x;
        diff_y = laser_cloud->points[i + 1].y * depth1 / depth2 - laser_cloud->points[i].y;
        diff_z = laser_cloud->points[i + 1].z * depth1 / depth2 - laser_cloud->points[i].z;

        if (sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) / depth1 < 0.1)
        {
          cloud_neighbor_picked_[i + 1] = 1;
          cloud_neighbor_picked_[i + 2] = 1;
          cloud_neighbor_picked_[i + 3] = 1;
          cloud_neighbor_picked_[i + 4] = 1;
          cloud_neighbor_picked_[i + 5] = 1;
          cloud_neighbor_picked_[i + 6] = 1;
        }
      }
    }

    //针对paper中(a)情况
    float diff_x_2 = laser_cloud->points[i].x - laser_cloud->points[i - 1].x;
    float diff_y_2 = laser_cloud->points[i].y - laser_cloud->points[i - 1].y;
    float diff_z_2 = laser_cloud->points[i].z - laser_cloud->points[i - 1].z;
    float diff_2 = diff_x_2 * diff_x_2 + diff_y_2 * diff_y_2 + diff_z_2 * diff_z_2;

    float dis = laser_cloud->points[i].x * laser_cloud->points[i].x + laser_cloud->points[i].y * laser_cloud->points[i].y + laser_cloud->points[i].z * laser_cloud->points[i].z;

    if (diff > 0.0002 * dis && diff_2 > 0.0002 * dis)
    {
      cloud_neighbor_picked_[i] = 1;
    }
  }
}

void FeatureDt::detectFeaturePoint(pcl::PointCloud<PointType>::Ptr laser_cloud,
                                  int cloud_size,
                                  std::vector<int> &scan_start_ind,
                                  std::vector<int> &scan_end_ind,
                                  pcl::PointCloud<PointType>::Ptr &corner_sharp,
                                  pcl::PointCloud<PointType>::Ptr &corner_less_sharp,
                                  pcl::PointCloud<PointType>::Ptr &surf_flat,
                                  pcl::PointCloud<PointType>::Ptr &surf_less_flat)
{
  //还是每束scan单独处理
  for (int i = 0; i < N_SCANS; i++)
  {
    pcl::PointCloud<PointType>::Ptr surf_points_less_flat_scan(new pcl::PointCloud<PointType>);
    int less_sharp_num = 0;

    //将每个线等分为六段，分别进行处理（sp、ep分别为各段的起始和终止位置）
    for (int j = 0; j < 6; j++)
    {
      //先求每段的开始和结束点
      int sp = (scan_start_ind[i] * (6 - j) + scan_end_ind[i] * j) / 6;
      int ep = (scan_start_ind[i] * (5 - j) + scan_end_ind[i] * (j + 1)) / 6 - 1;

      //在每一段，排序, 小的在前, 大的在后
      for (int k = sp + 1; k <= ep; k++)
      {
        for (int l = k; l >= sp + 1; l--)
        {
          if (cloud_curvature_[cloud_sort_ind_[l]] < cloud_curvature_[cloud_sort_ind_[l - 1]])
          {
            swap(cloud_sort_ind_[l - 1], cloud_sort_ind_[l]);
          }
        }
      }

      //选取角点
      int largest_picked_num = 0;
      for (int k = ep; k >= sp; k--)
      {
        //k = ep, cloud_sort_ind_[k]对应这一组最末位置的index, 也就是曲率最大的index
        int ind = cloud_sort_ind_[k];

        //如果邻居没被选中并且自己够格
        if (cloud_neighbor_picked_[ind] == 0 && cloud_curvature_[ind] > 0.1)
        {
          largest_picked_num++;

          //取x个认为是sharp的点
          if (largest_picked_num <= 6)
          {
            cloud_label_[ind] = CURV_SHARP;
            corner_sharp->push_back(laser_cloud->points[ind]);
            corner_less_sharp->push_back(laser_cloud->points[ind]);
            less_sharp_num++;
          }
          //取y个认为是lesssharp的点
          else if (largest_picked_num <= 24)
          {
            cloud_label_[ind] = CURV_LESS_SHARP;
            corner_less_sharp->push_back(laser_cloud->points[ind]);
            less_sharp_num++;
          }
          else
          {
            break;
          }

          //选中的点标记为已选
          cloud_neighbor_picked_[ind] = 1;

          //向后五个点
          for (int l = 1; l <= 5; l++)
          {
            //之前的曲率是前后各五个点, 这里计算相邻两点的变化率
            float diff_x = laser_cloud->points[ind + l].x - laser_cloud->points[ind + l - 1].x;
            float diff_y = laser_cloud->points[ind + l].y - laser_cloud->points[ind + l - 1].y;
            float diff_z = laser_cloud->points[ind + l].z - laser_cloud->points[ind + l - 1].z;

            //遇到某点曲率高不标记, 还有机会被选上
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
            {
              break;
            }

            //否则, 标记, 因为邻居是角点, 你不能再做角点
            cloud_neighbor_picked_[ind + l] = 1;
          }

          //向前五个点, 逻辑用上
          for (int l = -1; l >= -5; l--)
          {
            float diff_x = laser_cloud->points[ind + l].x - laser_cloud->points[ind + l + 1].x;
            float diff_y = laser_cloud->points[ind + l].y - laser_cloud->points[ind + l + 1].y;
            float diff_z = laser_cloud->points[ind + l].z - laser_cloud->points[ind + l + 1].z;

            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
            {
              break;
            }

            cloud_neighbor_picked_[ind + l] = 1;
          }
        }
      }

      //选取平面点
      int smallest_picked_num = 0;
      for (int k = sp; k <= ep; k++)
      {
        //!! k = sp, cloud_sort_ind_[k]对应这一组最先位置的index, 也就是曲率最小的index
        int ind = cloud_sort_ind_[k];
        if (cloud_neighbor_picked_[ind] == 0 && cloud_curvature_[ind] < 0.1)
        {
          cloud_label_[ind] = CURV_SMOOTH;
          surf_flat->push_back(laser_cloud->points[ind]);

          smallest_picked_num++;
          if (smallest_picked_num >= 8)
          {
            break;
          }

          //已选中的点, 对临近点进行标记
          cloud_neighbor_picked_[ind] = 1;

          //向后遍历五个点
          for (int l = 1; l <= 5; l++)
          {
            float diff_x = laser_cloud->points[ind + l].x - laser_cloud->points[ind + l - 1].x;
            float diff_y = laser_cloud->points[ind + l].y - laser_cloud->points[ind + l - 1].y;
            float diff_z = laser_cloud->points[ind + l].z - laser_cloud->points[ind + l - 1].z;

            //此处发生突变, 停止标记临近点
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
            {
              break;
            }

            cloud_neighbor_picked_[ind + l] = 1;
          }

          //向前遍历五个点, 逻辑同上
          for (int l = -1; l >= -5; l--)
          {
            float diff_x = laser_cloud->points[ind + l].x - laser_cloud->points[ind + l + 1].x;
            float diff_y = laser_cloud->points[ind + l].y - laser_cloud->points[ind + l + 1].y;
            float diff_z = laser_cloud->points[ind + l].z - laser_cloud->points[ind + l + 1].z;

            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
            {
              break;
            }

            cloud_neighbor_picked_[ind + l] = 1;
          }
        }
      }

      //取平面点 less smooth, 之前没被选中的都会被标记为less smooth
      for (int k = sp; k <= ep; k++)
      {
        // <= CURV_LESS_SMOOTH means smooth or less smooth
        if (cloud_label_[k] <= CURV_LESS_SMOOTH)
        {
          surf_points_less_flat_scan->push_back(laser_cloud->points[k]);
        }
      }
    }

    // 对lessFlatScan进行降采样
    pcl::PointCloud<PointType> surf_points_less_flat_scan_ds;
    pcl::VoxelGrid<PointType> down_size_filter;
    down_size_filter.setInputCloud(surf_points_less_flat_scan);
    down_size_filter.setLeafSize(0.15, 0.15, 0.15);
    down_size_filter.filter(surf_points_less_flat_scan_ds);

    //sp 是个step, 这里使用了一种简单的点过滤方法
    int sp = 1;

    if (less_sharp_num == 0)
    {
      sp = floor(1.0 * surf_points_less_flat_scan_ds.size() / 100);
    }
    else
    {
      sp = floor(1.0 * surf_points_less_flat_scan_ds.size() / less_sharp_num / 3);
    }

    sp = sp > 0 ? sp : 1;
    for (int k = 0; k < surf_points_less_flat_scan_ds.size(); k += sp)
    {
      surf_less_flat->push_back(surf_points_less_flat_scan_ds.points[k]);
    }
  }
}

int FeatureDt::getScanIDOfPoint(PointType &point)
{
  //线与水平面的夹角,计算线束的角度, 以确定属于哪条线, 单位 °
  float scan_pitch = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;

  //根据scanPitch确定scan ID, 范围0~N_SCANS - 1, scanID在后面寻找最近直线时将发挥重要作用
  int rounded_scan_pitch = int(scan_pitch / 2.0 + (scan_pitch < 0.0 ? -0.5 : +0.5));
  int scan_id = 0;

  //让SCANID在物理上连续
  scan_id = rounded_scan_pitch + 7;

  return scan_id;
}
