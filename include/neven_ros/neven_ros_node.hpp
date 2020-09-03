/*
 * Object segmentation via PCL Euclidean Cluster Extraction ROS node
 * neven_ros_node.hpp
 *
 *    _  _______   _______  __
 *   / |/ / __/ | / / __/ |/ /
 *  /    / _/ | |/ / _//    / 
 * /_/|_/___/ |___/___/_/|_/  
 *                            
 *
 * Copyright (C) 2020 Neven 
 * By 1487Quantum (https://github.com/1487quantum)
 * 
 * 
 * Licensed under the BSD-2-Clause License .
 * 
 */

#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>

#include <std_msgs/ColorRGBA.h>

class NevenNode {
public:
NevenNode(const ros::NodeHandle& nh_);
void init();
//Callback
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& pcl_in);

private:
ros::Subscriber subPCL;
ros::Publisher pubObjSeg;
ros::Publisher pubMarker;
ros::Publisher pubBoundRect;

    ros::NodeHandle nh; //Node handle

bool colorCluster{false};

//Utilities
geometry_msgs::Pose createPose(const float& sx, const float& sy, const float& sz);
geometry_msgs::Vector3 createScale(const float& sx, const float& sy, const float& sz);
std_msgs::ColorRGBA createColor(const float& r, const float& g, const float& b, const float& a);
//Visualisation
visualization_msgs::Marker addMarker(const geometry_msgs::Pose& mkr_pose, const geometry_msgs::Vector3& mkr_size, const std_msgs::ColorRGBA& mkr_clr, const float& idx, const char* mkr_ns, const int& mType);
visualization_msgs::Marker addCentroid(const float& pt_x, const float& pt_y, const float& pt_z, const float& idx);
visualization_msgs::Marker addBoundRect(const float& pt_x, const float& pt_y, const float& pt_z, const float& pt_l, const float& pt_b, const float& pt_h, const float& idx);
//PCL filters
void vx_grid(const pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out, const float& leafSize);
void sor_filter(const pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out, const int& mean_k, const float& std_thres);
void p_seg(pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out);
void euclus(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cld_in, const int &clus_min, const int &clus_max);
};
