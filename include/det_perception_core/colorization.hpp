/**
 * @file colorization.hpp
 * @author Jiasen Zheng 
 * @brief colorize the raw point cloud with the color image
 */

#ifndef DET_PERCEPTION_CORE_COLORIZATION_HPP
#define DET_PERCEPTION_CORE_COLORIZATION_HPP

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

class Colorization
{
public:
    Colorization(ros::NodeHandle nh);
    ~Colorization();
    void run();
    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_pointcloud;
    ros::Subscriber m_sub_image;
    ros::Publisher m_pub_pointcloud;
};

# endif  // DET_PERCEPTION_CORE_COLORIZATION_HPP
