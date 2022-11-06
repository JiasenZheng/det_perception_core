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
    ROS_INFO_STREAM("Initializing perception core ...");
    ROS_INFO_STREAM("Detecting table ...");
    // wait for the first pointcloud message
    auto first_point_cloud_msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(m_lidar_topic, m_nh, 
    ros::Duration(30.0));
    if (first_point_cloud_msg == nullptr)
    {
        ROS_ERROR_STREAM("No pointcloud message received!");
        return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*first_point_cloud_msg, *cloud);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    planeSegmentation<pcl::PointXYZ>(cloud, 100, 0.1, inliers, coefficients);
    ROS_INFO_STREAM("Table detected!");
    // log inliers and coefficients
    ROS_INFO_STREAM("Inliers: " << inliers->indices.size());
    ROS_INFO_STREAM("Coefficients: " << coefficients->values[0] << " " << coefficients->values[1] << " " <<
    coefficients->values[2] << " " << coefficients->values[3]);
    m_plane_coefficients = coefficients;
}

void PerceptionCore::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{   
    if (m_plane_coefficients == nullptr)
    {
        return;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *cloud);
    removePlane<pcl::PointXYZRGB>(cloud, m_plane_coefficients, 0.01);
    sensor_msgs::PointCloud2 processed_cloud_msg;
    pcl::toROSMsg(*cloud, processed_cloud_msg);
    processed_cloud_msg.header.frame_id = msg->header.frame_id;
    processed_cloud_msg.header.stamp = msg->header.stamp;
    m_pointcloud_pub.publish(processed_cloud_msg);
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

template <typename T>
void PerceptionCore::removePlane(typename pcl::PointCloud<T>::Ptr cloud, const pcl::ModelCoefficients::Ptr coefficients,
const double& distance_threshold) {
    // compute the distance from each point to the plane
    for (size_t i = 0; i < cloud->points.size(); i++) {
        double distance = std::abs(coefficients->values[0] * cloud->points[i].x + coefficients->values[1] * 
        cloud->points[i].y + coefficients->values[2] * cloud->points[i].z + coefficients->values[3]) / 
        std::sqrt(coefficients->values[0] * coefficients->values[0] + coefficients->values[1] * 
        coefficients->values[1] + coefficients->values[2] * coefficients->values[2]);
        // remove the point if it is too close to the plane
        if (distance < distance_threshold) {
            cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
            cloud->points[i].y = std::numeric_limits<float>::quiet_NaN();
            cloud->points[i].z = std::numeric_limits<float>::quiet_NaN();
        }
        // // remove the point if it is below the plane
        if (cloud->points[i].z > std::abs(coefficients->values[3])) {
            cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
            cloud->points[i].y = std::numeric_limits<float>::quiet_NaN();
            cloud->points[i].z = std::numeric_limits<float>::quiet_NaN();
        }
    }
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
