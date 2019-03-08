// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#include "loam_velodyne/MultiScanRegistration.h"
#include "math_utils.h"

#include <pcl_conversions/pcl_conversions.h>


namespace loam {

MultiScanMapper::MultiScanMapper(const float& lowerBound,
                                 const float& upperBound,
                                 const uint16_t& nScanRings)
    : _lowerBound(lowerBound),
      _upperBound(upperBound),
      _nScanRings(nScanRings),
      _factor((nScanRings - 1) / (upperBound - lowerBound))
{

}

void MultiScanMapper::set(const float &lowerBound,
                          const float &upperBound,
                          const uint16_t &nScanRings)
{
  _lowerBound = lowerBound;
  _upperBound = upperBound;
  _nScanRings = nScanRings;
  _factor = (nScanRings - 1) / (upperBound - lowerBound);
}

void MultiScanMapper::debug_multiscanmapper()
{
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "debug_multiscanmapper:   " << std::endl;
  std::cout << "lowbound:   " << _lowerBound <<  "    factor:   " << _factor << std::endl;
}


int MultiScanMapper::getRingForAngle(const float& angle) {
  //debug_multiscanmapper();
  return int(((angle * 180 / M_PI) - _lowerBound) * _factor + 0.5);
}






MultiScanRegistration::MultiScanRegistration(const MultiScanMapper& scanMapper,
                                             const RegistrationParams& config)
    : ScanRegistration(config),
      _systemDelay(SYSTEM_DELAY),
      _scanMapper(scanMapper)
{

};



bool MultiScanRegistration::setup(ros::NodeHandle& node,
                                  ros::NodeHandle& privateNode)
{
  //这个setup函数里订阅imu消息，并且注册一些消息发布
  if (!ScanRegistration::setup(node, privateNode)) {
    return false;
  }

  // fetch scan mapping params
  std::string lidarName;

  if (privateNode.getParam("lidar", lidarName)) {
    if (lidarName == "VLP-16") {
      _scanMapper = MultiScanMapper::Velodyne_VLP_16();
    } else if (lidarName == "HDL-32") {
      _scanMapper = MultiScanMapper::Velodyne_HDL_32();
    } else if (lidarName == "HDL-64E") {
      _scanMapper = MultiScanMapper::Velodyne_HDL_64E();
    } else {
      ROS_ERROR("Invalid lidar parameter: %s (only \"VLP-16\", \"HDL-32\" and \"HDL-64E\" are supported)", lidarName.c_str());
      return false;
    }

    ROS_INFO("Set  %s  scan mapper.", lidarName.c_str());
    if (!privateNode.hasParam("scanPeriod")) {
      _config.scanPeriod = 0.1;
      ROS_INFO("Set scanPeriod: %f", _config.scanPeriod);
    }
  } else {
    float vAngleMin, vAngleMax;
    int nScanRings;

    if (privateNode.getParam("minVerticalAngle", vAngleMin) &&
        privateNode.getParam("maxVerticalAngle", vAngleMax) &&
        privateNode.getParam("nScanRings", nScanRings)) {
      if (vAngleMin >= vAngleMax) {
        ROS_ERROR("Invalid vertical range (min >= max)");
        return false;
      } else if (nScanRings < 2) {
        ROS_ERROR("Invalid number of scan rings (n < 2)");
        return false;
      }

      _scanMapper.set(vAngleMin, vAngleMax, nScanRings);
      ROS_INFO("Set linear scan mapper from %g to %g degrees with %d scan rings.", vAngleMin, vAngleMax, nScanRings);
    }
  }


  // subscribe to input cloud topic
  _subLaserCloud = node.subscribe<sensor_msgs::PointCloud2>
      ("/multi_scan_points", 2, &MultiScanRegistration::handleCloudMessage, this);

  return true;
}



void MultiScanRegistration::handleCloudMessage(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
  if (_systemDelay > 0) {
    _systemDelay--;
    return;
  }

  // fetch new input cloud
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

  process(laserCloudIn, laserCloudMsg->header.stamp);
}


//求出这个激光点云中每个点的pitch角,根据获得的pitch角度求得线序
//按照线序存储激光点云
//提取特征点
//发布相应的特征点云
void MultiScanRegistration::process(const pcl::PointCloud<pcl::PointXYZ>& laserCloudIn,
                                    const ros::Time& scanTime)
{
  size_t cloudSize = laserCloudIn.size();

  // reset internal buffers and set IMU start state based on current scan time
  reset(scanTime);

  // determine scan start and end orientations
  //lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
  float startOri = -std::atan2(laserCloudIn[0].y, laserCloudIn[0].x);
  float endOri = -std::atan2(laserCloudIn[cloudSize - 1].y,
                             laserCloudIn[cloudSize - 1].x) + 2 * float(M_PI);
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  bool halfPassed = false;
  pcl::PointXYZI point;
  std::vector<pcl::PointCloud<pcl::PointXYZI> > laserCloudScans(_scanMapper.getNumberOfScanRings());

  // extract valid points from input cloud
  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn[i].y;
    point.y = laserCloudIn[i].z;
    point.z = laserCloudIn[i].x;

    // skip NaN and INF valued points
    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    // skip zero valued points
    if (point.x * point.x + point.y * point.y + point.z * point.z < 0.0001) {
      continue;
    }

    // calculate vertical point angle and scan ID
    //通过计算pitch角来区分激光的线序
    float angle = std::atan(point.y / std::sqrt(point.x * point.x + point.z * point.z));
    int scanID = _scanMapper.getRingForAngle(angle);
    //std::cout << "debug   _scanMapper.getNumberOfScanRings():   " << _scanMapper.getNumberOfScanRings() << std::endl;
    //_scanMapper.getNumberOfScanRings()表示velodyne的激光束数量
    if (scanID >= _scanMapper.getNumberOfScanRings() || scanID < 0 ){
      continue;
    }

    // calculate horizontal point angle
    float ori = -std::atan2(point.x, point.z);
    if (!halfPassed) {
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      }
    }

    // calculate relative scan time based on point orientation
    //relTime表示的是相对时间，这个通过角度的比值获得
    float relTime = _config.scanPeriod * (ori - startOri) / (endOri - startOri);
    point.intensity = scanID + relTime;

    // project point to the start of the sweep using corresponding IMU data
    if (hasIMUData()) {
      setIMUTransformFor(relTime);
      //转换到global imu system是为什么，激光坐标系和imu坐标系之间的关系是怎样？
      transformToStartIMU(point);
    }
    //按照线序保存激光点云
    laserCloudScans[scanID].push_back(point);
  }

  // construct sorted full resolution cloud
  cloudSize = 0;
  for (int i = 0; i < _scanMapper.getNumberOfScanRings(); i++) {
    _laserCloud += laserCloudScans[i];

    IndexRange range(cloudSize, 0);
    cloudSize += laserCloudScans[i].size();
    //range表示每线激光数据的开始和结束序号
    range.second = cloudSize > 0 ? cloudSize - 1 : 0;
    //_scanIndices保存range，分别表示第0到第16线激光的开始和结束序号
    _scanIndices.push_back(range);
  }

  // extract features
  extractFeatures();

  // publish result
  publishResult();
}

} // end namespace loam
