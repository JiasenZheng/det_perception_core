/**
 * @file perception_core.cpp
 * @author Jiasen Zheng
 * @brief Core class for perception tasks
 */
 
#include "perception_core.hpp"

PerceptionCore::PerceptionCore(ros::NodeHandle nh): m_nh(nh)
{
    m_lidar_topic = "/l515/depth/color/points";
    m_pointcloud_sub = m_nh.subscribe(m_lidar_topic, 1, &PerceptionCore::pointcloudCallback, this);
    // m_image_sub = m_nh.subscribe("/l515/color/image_raw", 1, &PerceptionCore::imageCallback, this);
    m_pointcloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/l515/points/processed", 1);
}

void PerceptionCore::run()
{
    
}

void PerceptionCore::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    planeSegmentation<pcl::PointXYZ>(cloud, 100, 0.01, inliers, coefficients);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_colored = colorizeSegmentation<pcl::PointXYZ>(cloud, inliers);
    sensor_msgs::PointCloud2 processed_msg;
    pcl::toROSMsg(cloud_colored, processed_msg);
    processed_msg.header.frame_id = msg->header.frame_id;
    processed_msg.header.stamp = msg->header.stamp;
    m_pointcloud_pub.publish(processed_msg);
}

template <typename T>
void PerceptionCore::planeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud, int max_iterations, 
double distance_threshold, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients) {
    pcl::SACSegmentation<T> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations);
    seg.setDistanceThreshold(distance_threshold);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
}

template <typename T>
pcl::PointCloud<pcl::PointXYZRGB> PerceptionCore::colorizeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud, 
const pcl::PointIndices::Ptr inliers) {
    pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
    colored_cloud.width = cloud->width;
    colored_cloud.height = cloud->height;
    colored_cloud.is_dense = cloud->is_dense;
    colored_cloud.points.resize(cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); i++) {
        colored_cloud.points[i].x = cloud->points[i].x;
        colored_cloud.points[i].y = cloud->points[i].y;
        colored_cloud.points[i].z = cloud->points[i].z;
        if (std::find(inliers->indices.begin(), inliers->indices.end(), i) != inliers->indices.end()) {
            colored_cloud.points[i].r = 255;
            colored_cloud.points[i].g = 0;
            colored_cloud.points[i].b = 0;
        } else {
            colored_cloud.points[i].r = 0;
            colored_cloud.points[i].g = 255;
            colored_cloud.points[i].b = 0;
        }
    }
    return colored_cloud;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "perception_core");
    ros::NodeHandle nh;
    PerceptionCore perception_core(nh);
    perception_core.run();
    ros::spin();
    return 0;
}
