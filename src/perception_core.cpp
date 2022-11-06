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
    if (m_plane_coefficients == nullptr)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        planeSegmentation<pcl::PointXYZ>(cloud, 100, 0.1, inliers, coefficients);
        ROS_INFO_STREAM("Table detected!");
        m_plane_coefficients = coefficients;
        // log inliers and coefficients
        ROS_INFO_STREAM("Inliers: " << inliers->indices.size());
        ROS_INFO_STREAM("Coefficients: " << coefficients->values[0] << " " << coefficients->values[1] << " " <<
        coefficients->values[2] << " " << coefficients->values[3]);
        m_plane_coefficients = coefficients;
        return;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *cloud);
    auto cloud_filtered = removePlane<pcl::PointXYZRGB>(cloud, m_plane_coefficients, 0.01);
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_filtered, cloud_msg);
    cloud_msg.header.frame_id = msg->header.frame_id;
    cloud_msg.header.stamp = msg->header.stamp;
    m_pointcloud_pub.publish(cloud_msg);
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
typename pcl::PointCloud<T>::Ptr PerceptionCore::removePlane(const typename pcl::PointCloud<T>::Ptr cloud, 
const pcl::ModelCoefficients::Ptr coefficients, const double& distance_threshold) {
    typename pcl::PointCloud<T>::Ptr cloud_filtered(new pcl::PointCloud<T>);
    //compute the distance from each point to the plane
    for (size_t i = 0; i < cloud->points.size(); i++) {
        double distance = std::abs(coefficients->values[0] * cloud->points[i].x + coefficients->values[1] * 
        cloud->points[i].y + coefficients->values[2] * cloud->points[i].z + coefficients->values[3]) / 
        std::sqrt(coefficients->values[0] * coefficients->values[0] + coefficients->values[1] * 
        coefficients->values[1] + coefficients->values[2] * coefficients->values[2]);
        //remove the point if it is too close to the plane
        if (distance < distance_threshold) {
            continue;
        }
        // remove the point if it is below the plane
        if (cloud->points[i].z > std::abs(coefficients->values[3])) {
            continue;
        }
        cloud_filtered->points.push_back(cloud->points[i]);
    }
    cloud_filtered->width = cloud_filtered->points.size();
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = true;
    return cloud_filtered;
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
