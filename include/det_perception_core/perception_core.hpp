/**
 * @file perception_core.hpp
 * @author Jiasen Zheng
 * @brief Core class for perception tasks
 */
 
#pragma once

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

class PerceptionCore
{
public:
    PerceptionCore(ros::NodeHandle nh);
    void run();
    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    template <typename T>
    void planeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud, int max_iterations, 
    double distance_threshold, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
    template <typename T>
    pcl::PointCloud<pcl::PointXYZRGB> colorizeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud, 
    const pcl::PointIndices::Ptr inliers);

private:
    ros::NodeHandle m_nh;
    std::string m_lidar_topic;
    ros::Subscriber m_pointcloud_sub;
    ros::Subscriber m_image_sub;
    ros::Publisher m_pointcloud_pub;
};
