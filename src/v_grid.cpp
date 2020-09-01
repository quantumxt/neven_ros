#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
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

ros::Publisher pubObjSeg;
ros::Publisher pubMarker;

visualization_msgs::Marker addMarker(const float& pt_x, const float& pt_y, const float& pt_z, const float& idx, const char * mkr_ns, const int& mType)
{

    visualization_msgs::Marker marker;
    marker.header.frame_id = "cam_link";
    marker.header.stamp = ros::Time();
    marker.ns = mkr_ns;
    marker.id = idx;
    marker.type = mType;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = pt_x;
    marker.pose.position.y = pt_y;
    marker.pose.position.z = pt_z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.lifetime = ros::Duration(5.0); //Expires in 5s
    return marker;
}

visualization_msgs::Marker addCentroid(const float& pt_x, const float& pt_y, const float& pt_z, const float& idx)
{
    return addMarker(pt_x, pt_y, pt_z, idx, "pcl_marker_centroid", visualization_msgs::Marker::SPHERE);
}

//Downsample/Reduces num of pcl
void vx_grid(const pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out, const float &leafSize)
{
    //    ROS_INFO("Running Voxel Filter...");
    constexpr float f_limit[2] = { 0.01f, 1.8f };
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
void sor_filter(const pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out, const int &mean_k, const float &std_thres)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;

    sor.setInputCloud(cld_in.makeShared());
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_thres);
    sor.filter(cld_out);
    pcl::io::savePCDFileASCII("pcl_sor.pcd", cld_out);
}

void p_seg(pcl::PointCloud<pcl::PointXYZRGBA>& cld_in, pcl::PointCloud<pcl::PointXYZRGBA>& cld_out)
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

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& pcl_in)
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

    vx_grid(cloud, cloud_filtered,  0.005f); 		 //Voxel filter, 0.5cm leafsize
    sor_filter(cloud_filtered, cloud_filtered2, 50, 0.15f); //StatisticalOutlierRemoval Filter, mean_k:50, std_dv_threshold: 0.15f
    //p_seg(cloud_filtered2,cloud_filtered3);	 //Segmentation

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
    tree->setInputCloud(cloud_filtered2.makeShared());

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
    ec.setClusterTolerance(0.02); // 2cm
    ec.setMinClusterSize(150);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered2.makeShared());
    ec.extract(cluster_indices);

    ROS_INFO("Clusters: %i", cluster_indices.size());

    visualization_msgs::MarkerArray m_a;

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            cloud_cluster->push_back((cloud_filtered2)[*pit]); //*
            //ROS_INFO("XX [%i]: %i",j, *pit);
            cloud_filtered.points[*pit].r = 0;
            cloud_filtered2.points[*pit].g = 40 * (j);
            cloud_filtered2.points[*pit].b = 0;
            cloud_filtered2.points[*pit].a = 255;
        }
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        ROS_INFO("Cluster [%i]: %i", j, cloud_cluster->size());

        //Calculate centroid
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_cluster, centroid);
        ROS_INFO("Centroid [%i]: %f %f %f %f", j, centroid[0], centroid[1], centroid[2], centroid[3]);

        m_a.markers.push_back(addCentroid(centroid[0], centroid[1], centroid[2], j));

        pubObjSeg.publish(cloud_filtered2);

        ++j;
    }
    pcl::io::savePCDFileASCII("save_3.pcd", cloud_filtered2);
    pubMarker.publish(m_a);
}

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "v_grid");
    ros::NodeHandle nh;

    //ros::param::set("dist_th", 0.1);	// Set ROS param

    ros::Subscriber sub = nh.subscribe("/cam/points2", 1, cloud_cb);	// Create a ROS subscriber for the input point cloud

    // Create a ROS publisher for the model coefficients
    pubObjSeg = nh.advertise<pcl::PointCloud<pcl::PointXYZRGBA> >("obj_segmentation", 1);
    pubMarker = nh.advertise<visualization_msgs::MarkerArray>("obj_centroid", 1);

    // Spin
    ros::spin();
}
