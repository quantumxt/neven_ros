/*
 * Object segmentation via PCL Euclidean Cluster Extraction ROS node
 * neven_ros_node.cpp
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

#include "neven_ros/neven_ros_node.hpp"

NevenNode::NevenNode(const ros::NodeHandle& nh_){
    this->nh = nh_;
}

void NevenNode::init(){
 ros::Subscriber sub = this->nh.subscribe("/cam/points2", 1, &NevenNode::cloud_cb, this); // ROS subscriber

    //ROS publisher
    pubObjSeg = this->nh.advertise<pcl::PointCloud<pcl::PointXYZRGBA> >("obj_segmentation", 1);
    pubMarker = this->nh.advertise<visualization_msgs::MarkerArray>("obj_centroid", 1);
    pubBoundRect = this->nh.advertise<visualization_msgs::MarkerArray>("obj_boundRect", 1);
}

void NevenNode::initMinMax(minMax& src)
{
    constexpr float pMin{ 10000.0f };
    src.pt_min.x = pMin;
    src.pt_min.y = pMin;
    src.pt_min.z = pMin;
    src.pt_max.x = -pMin;
    src.pt_max.y = -pMin;
    src.pt_max.z = -pMin;
}

void NevenNode::compareVals(const pcl::PointXYZRGBA& pcl_in, minMax& src)
{
    if (pcl_in.x < src.pt_min.x) {
        src.pt_min.x = pcl_in.x;
    }
    else if (pcl_in.x > src.pt_max.x) {
        src.pt_max.x = pcl_in.x;
    }

    if (pcl_in.y < src.pt_min.y) {
        src.pt_min.y = pcl_in.y;
    }
    else if (pcl_in.y > src.pt_max.y) {
        src.pt_max.y = pcl_in.y;
    }

    if (pcl_in.z < src.pt_min.z) {
        src.pt_min.z = pcl_in.z;
    }
    else if (pcl_in.z > src.pt_max.z) {
        src.pt_max.z = pcl_in.z;
    }

    //ROS_INFO("MinMax [x y z x y z]: %f %f %f %f %f %f", src.pt_min.x,src.pt_min.y,src.pt_min.z, src.pt_max.x,src.pt_max.y,src.pt_max.z);
}

geometry_msgs::Pose NevenNode::createPose(const float& sx, const float& sy, const float& sz)
{
    geometry_msgs::Pose ps;
    ps.position.x = sx;
    ps.position.y = sy;
    ps.position.z = sz;
    ps.orientation.x = 0.0;
    ps.orientation.y = 0.0;
    ps.orientation.z = 0.0;
    ps.orientation.w = 1.0;
    return ps;
}

geometry_msgs::Vector3 NevenNode::createScale(const float& sx, const float& sy, const float& sz)
{
    geometry_msgs::Vector3 scl;
    scl.x = sx;
    scl.y = sy;
    scl.z = sz;
    return scl;
}

std_msgs::ColorRGBA NevenNode::createColor(const float& r, const float& g, const float& b, const float& a)
{
    std_msgs::ColorRGBA mClr;
    mClr.r = r;
    mClr.g = g;
    mClr.b = b;
    mClr.a = a; // Don't forget to set the alpha! (0.0-1.0)
    return mClr;
}

visualization_msgs::Marker NevenNode::addMarker(const geometry_msgs::Pose& mkr_pose, const geometry_msgs::Vector3& mkr_size, const std_msgs::ColorRGBA& mkr_clr, const float& idx, const char* mkr_ns, const int& mType)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "cam_link";
    marker.header.stamp = ros::Time();
    marker.ns = mkr_ns;
    marker.id = idx;
    marker.type = mType;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose = mkr_pose;
    marker.scale = mkr_size;
    marker.color = mkr_clr;
    marker.lifetime = ros::Duration(5.0); //Expires in 5s
    return marker;
}

visualization_msgs::Marker NevenNode::addCentroid(const float& pt_x, const float& pt_y, const float& pt_z, const float& idx)
{
    return addMarker(createPose(pt_x, pt_y, pt_z), createScale(0.05, 0.05, 0.05), createColor(0.0, 1.0, 0.0, 1.0), idx, "pcl_marker_centroid", visualization_msgs::Marker::SPHERE);
}

visualization_msgs::Marker NevenNode::addBoundRect(const float& pt_x, const float& pt_y, const float& pt_z, const float& pt_l, const float& pt_b, const float& pt_h, const float& idx)
{
    return addMarker(createPose(pt_x, pt_y, pt_z), createScale(pt_l, pt_b, pt_h), createColor(0.0, 0.0, 1.0, 0.5), idx, "pcl_marker_boundRect", visualization_msgs::Marker::CUBE);
}

//Downsample/Reduces num of pcl
void NevenNode::vx_grid(const pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out, const float& leafSize)
{
    //    ROS_INFO("Running Voxel Filter...");
    constexpr float f_limit[2] = { 0.01f, 3.0f };
    pcl::VoxelGrid<pcl::PointXYZRGBA> vxg;

    vxg.setInputCloud(cld_in.makeShared());
    vxg.setLeafSize(leafSize, leafSize, leafSize);
    vxg.setFilterLimits(f_limit[0], f_limit[1]);
    vxg.setFilterFieldName("z");
    vxg.setFilterLimitsNegative(false);
    vxg.filter(cld_out);
    pcl::io::savePCDFileASCII("pcl_vxg.pcd", cld_out);
}

//Filters noise
void NevenNode::sor_filter(const pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out, const int& mean_k, const float& std_thres)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
    sor.setInputCloud(cld_in.makeShared());
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_thres);
    sor.filter(cld_out);
    pcl::io::savePCDFileASCII("pcl_sor.pcd", cld_out);
}

void NevenNode::p_seg(pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out)
{
    //Segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::PointCloud<pcl::PointXYZRGBA> cloud_tmp; //Temp pcl

    pcl::SACSegmentation<pcl::PointXYZRGBA> seg;

    seg.setOptimizeCoefficients(true); // Optional
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01);

    int i = 0, nr_points = (int)cld_in.size();

    //Planar segmentation
    // While 30% of the original cloud is still there
    while (cld_in.size() > 0.3 * nr_points) {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud(cld_in.makeShared());
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            ROS_INFO("Could not estimate a planar model for the given dataset.");
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
        extract.setInputCloud(cld_in.makeShared());
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(cld_out);

        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        extract.filter(cloud_tmp);
        cld_in.swap(cloud_tmp);

        i++;
    }
}

void NevenNode::euclus(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cld_in)
{
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
    tree->setInputCloud(cld_in);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
    ec.setClusterTolerance(0.02); // 2cm
    ec.setMinClusterSize(150);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cld_in);
    ec.extract(cluster_indices);

    ROS_INFO("Clusters: %i", cluster_indices.size());

    //Markers
    visualization_msgs::MarkerArray m_centroid;
    visualization_msgs::MarkerArray m_boundRect;

    int j = 0; //track clusters

    minMax bound_points; //Used to keep track of the bounding volume of the pcl cluster
    initMinMax(bound_points);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cHull_pts(new pcl::PointCloud<pcl::PointXYZRGBA>); //Temp pcl to hold convex hull pts

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            cloud_cluster->push_back((*cld_in)[*pit]); //*
            //ROS_INFO("XX [%i]: %i",j, *pit);
            cld_in->points[*pit].r = 0;
            cld_in->points[*pit].g = 40 * (j + 1);
            cld_in->points[*pit].b = 0;
            cld_in->points[*pit].a = 255;
        }

        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        //ROS_INFO("Cluster [%i]: %i", j, cloud_cluster->size());

        //Calculate centroid
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_cluster, centroid);
        //   ROS_INFO("Centroid [%i]: %f %f %f %f", j, centroid[0], centroid[1], centroid[2], centroid[3]);
        m_centroid.markers.push_back(addCentroid(centroid[0], centroid[1], centroid[2], j));

        // ROS_INFO("Final MinMax [x y z x y z]: %f %f %f %f %f %f", bound_points.pt_min.x, bound_points.pt_min.y, bound_points.pt_min.z, bound_points.pt_max.x, bound_points.pt_max.y, bound_points.pt_max.z);

        //Get convex hull to get minmax
        pcl::ConvexHull<pcl::PointXYZRGBA> cHull;
        cHull.setInputCloud(cloud_cluster);
        cHull.reconstruct(*cHull_pts);

        for (int k{ 0 }; k < cHull_pts->size(); ++k) {
            compareVals(cHull_pts->points[k], bound_points); //Get minMax vals of cluster
        }

        float length{ bound_points.pt_max.x - bound_points.pt_min.x };
        float breadth{ bound_points.pt_max.y - bound_points.pt_min.y };
        float height{ bound_points.pt_max.z - bound_points.pt_min.z };
        m_boundRect.markers.push_back(addBoundRect(centroid[0], centroid[1], centroid[2], length, breadth, height, j));
        //      ROS_INFO("Final lbh [x y z]: %f %f %f ", length, breadth, height);

        pubObjSeg.publish(*cld_in);

        ++j;
        initMinMax(bound_points);
    }
    pcl::io::savePCDFileASCII("save_3.pcd", *cld_in);

    pubMarker.publish(m_centroid);
    pubBoundRect.publish(m_boundRect);
}

void NevenNode::cloud_cb(const sensor_msgs::PointCloud2ConstPtr& pcl_in)
{
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZRGBA> cloud;
    pcl::PointCloud<pcl::PointXYZRGBA> cloud_filtered, cloud_filtered2, cloud_filtered3;
    pcl::fromROSMsg(*pcl_in, cloud);

    //ROS_INFO("Running CB...");

    // setting alpha = 1.0
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        cloud.points[i].a = 255;
    }

    vx_grid(cloud, cloud_filtered, 0.005f); //Voxel filter, 0.5cm leafsize
    sor_filter(cloud_filtered, cloud_filtered2, 50, 0.15f); //StatisticalOutlierRemoval Filter, mean_k:50, std_dv_threshold: 0.15f
    //p_seg(cloud_filtered2,cloud_filtered3);	 //Segmentation 	//Cpommented out as stereo image does not show any ground plane

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cld_tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
    *cld_tmp = cloud_filtered2;
    euclus(cld_tmp);
}

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "neven_node");
    ros::NodeHandle nh;

    //ros::param::set("dist_th", 0.1);	// Set ROS param

NevenNode nvn(nh);
nvn.init();
   
    // Spin
    ros::spin();
}
