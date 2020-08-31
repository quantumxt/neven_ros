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

ros::Publisher pub;
ros::Publisher pubMarker;

visualization_msgs::Marker addMarker(const float& pt_x, const float& pt_y, const float& pt_z, const float& idx)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "cam_link";
    marker.header.stamp = ros::Time();
    marker.ns = "pcl_marker_centroid";
    marker.id = idx;
    marker.type = visualization_msgs::Marker::SPHERE;
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

//Downsample/Reduces num of pcl
void vx_grid(const pcl::PointCloud<pcl::PointXYZRGBA> &cld_in, pcl::PointCloud<pcl::PointXYZRGBA> &cld_out){
//    ROS_INFO("Running Voxel Filter...");
 constexpr float leafSize{ 0.005f }; //Set it to 0.5cm
constexpr float f_limit[2] = {0.01f, 1.8f};
    pcl::VoxelGrid<pcl::PointXYZRGBA> vxg;

    vxg.setInputCloud(cld_in.makeShared());
    vxg.setLeafSize(leafSize, leafSize, leafSize);
    vxg.setFilterLimits(f_limit[0],f_limit[1]);
    vxg.setFilterFieldName("z");
    vxg.setFilterLimitsNegative(false);
    vxg.filter(cld_out);
    pcl::io::savePCDFileASCII("save_1.pcd", cld_out);
}

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& pcl_in)
{
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZRGBA> cloud;
    pcl::PointCloud<pcl::PointXYZRGBA> cloud_filtered, cloud_filtered2, cloud_filtered3, cloud_tmp;
    pcl::fromROSMsg(*pcl_in, cloud);

    ROS_INFO("Running CB...");

    // setting alpha = 1.0
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        cloud.points[i].a = 255;
    }

   

vx_grid(cloud,cloud_filtered);  //Voxel filter
   

    //StatisticalOutlierRemoval Filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
    sor.setInputCloud(cloud_filtered.makeShared());
    sor.setMeanK(50);
    sor.setStddevMulThresh(0.15);
    sor.filter(cloud_filtered2);
    pcl::io::savePCDFileASCII("save_2.pcd", cloud_filtered2);

    //Segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::SACSegmentation<pcl::PointXYZRGBA> seg;

    seg.setOptimizeCoefficients(true); // Optional
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01);

    int i = 0, nr_points = (int)cloud_filtered2.size();
    /*
//Planar segmentation
  // While 30% of the original cloud is still there
  while (cloud_filtered2.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(cloud_filtered2.makeShared());
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      ROS_INFO("Could not estimate a planar model for the given dataset.");
      break;
    }

    // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
    extract.setInputCloud(cloud_filtered2.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(cloud_filtered3);

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter(cloud_tmp);
    cloud_filtered2.swap (cloud_tmp);

    i++;
  }
*/

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
            cloud_filtered.points[*pit].r = 255;
            cloud_filtered2.points[*pit].g = 40 * (j);
            cloud_filtered2.points[*pit].b = 255 - 40 * j;
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

        m_a.markers.push_back(addMarker(centroid[0], centroid[1], centroid[2], j));

        pub.publish(cloud_filtered2);

        j++;
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

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe("/cam/points2", 1, cloud_cb);

    // Create a ROS publisher for the model coefficients
    pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGBA> >("obj_segmentation", 1);
    pubMarker = nh.advertise<visualization_msgs::MarkerArray>("obj_marker", 1);

    // Spin
    ros::spin();
}
