/**
 * @file colorization.hpp
 * @author Jiasen Zheng 
 * @brief colorize the raw point cloud with the color image
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

class Colorization
{
public:
    Colorization(ros::NodeHandle nh);
    void run();
    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void parseMatrix(const std::string& path, cv::Mat& matrix);
    void colorizeCloud(const cv::Mat& image, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colorized_cloud, const cv::Mat& intrinsic, const cv::Mat& extrinsic);


private:
    ros::NodeHandle m_nh;
    std::string m_cam_name;
    std::string m_lidar_name;
    ros::Subscriber m_sub_pointcloud;
    ros::Subscriber m_sub_image;
    ros::Publisher m_pub_pointcloud;
    cv::Mat m_image;
    cv::Mat m_intrinsic;
    cv::Mat m_extrinsic;
};
