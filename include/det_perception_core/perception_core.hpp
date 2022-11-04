/**
 * @file perception_core.hpp
 * @author Jiasen Zheng
 * @brief Core class for perception tasks
 */
 
#pragma once

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

class PerceptionCore
{
public:
    PerceptionCore(ros::NodeHandle nh);
    void run();
    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
private:
    ros::NodeHandle m_nh;
    std::string m_lidar_topic;
    ros::Subscriber m_pointcloud_sub;
    ros::Subscriber m_image_sub;
    ros::Publisher m_pointcloud_pub;
};
