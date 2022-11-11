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
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// data structure to store ordered point cloud and its start pixel coordinate
template <typename T>
struct OrderedCloud {
    // constructor
    OrderedCloud() : start_x(0), start_y(0) {
        // initialize point cloud
        cloud.reset(new pcl::PointCloud<T>);
    };
    OrderedCloud(typename pcl::PointCloud<T>::Ptr cloud, int x, int y) : cloud(cloud), start_x(x), start_y(y) {};
    // data
    typename pcl::PointCloud<T>::Ptr cloud;
    int start_x;
    int start_y;
    // create a pointer
    typedef boost::shared_ptr<OrderedCloud<T>> Ptr;
};



class PerceptionCore
{
public:
    PerceptionCore(ros::NodeHandle nh);
    void run();
    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    template <typename T>
    typename OrderedCloud<T>::Ptr shrinkOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud);
    template <typename T>
    void planeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud, int max_iterations, 
    double distance_threshold, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
    template <typename T>
    pcl::PointCloud<pcl::PointXYZRGB> colorizeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud, 
    const pcl::PointIndices::Ptr inliers);
    template <typename T>
    typename pcl::PointCloud<T>::Ptr removePlane(const typename pcl::PointCloud<T>::Ptr cloud, 
    const pcl::ModelCoefficients::Ptr coefficients, const double& distance_threshold);
    template <typename T>
    typename pcl::PointCloud<T>::Ptr cropOrderedCloud(const typename pcl::PointCloud<T>::Ptr cloud, 
    const int& margin_pixels);
    template <typename T>
    typename OrderedCloud<T>::Ptr cropOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
    const int& margin_pixels);
    template <typename T>
    typename pcl::PointCloud<T>::Ptr cropOrderedCloud(const typename pcl::PointCloud<T>::Ptr cloud,
    const int& left_pixels, const int& right_pixels, const int& top_pixels, const int& bottom_pixels);
    template <typename T>
    typename OrderedCloud<T>::Ptr cropOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
    const int& left_pixels, const int& right_pixels, const int& top_pixels, const int& bottom_pixels);
    cv::Mat imageBackgroundSubtraction(const cv::Mat& image, const cv::Mat& background, const int& threshold);
    template <typename T>
    typename pcl::PointCloud<T>::Ptr removePlane(const typename pcl::PointCloud<T>::Ptr cloud, 
    const cv::Mat& foreground_mask, const pcl::ModelCoefficients::Ptr coefficients, const double& distance_threshold);
    template <typename T>
    typename OrderedCloud<T>::Ptr removePlane(const typename OrderedCloud<T>::Ptr ordered_cloud,
    const cv::Mat& foreground_mask, const pcl::ModelCoefficients::Ptr coefficients, const double& distance_threshold);
    template <typename T>
    void getPlaneLimits(const typename pcl::PointCloud<T>::Ptr cloud, const pcl::PointIndices::Ptr inliers, 
    std::vector<double>& limits);

private:
    int m_image_count;
    int m_margin_pixels;
    std::vector<double> m_plane_limits;
    cv::Mat m_background_image;
    cv::Mat m_foreground_mask;
    std::string m_background_image_path;
    ros::NodeHandle m_nh;
    std::string m_lidar_topic;
    ros::Subscriber m_pointcloud_sub;
    ros::Subscriber m_image_sub;
    ros::Publisher m_cropped_cloud_pub;
    ros::Publisher m_processed_cloud_pub;
    ros::Publisher m_processed_image_pub;
    pcl::ModelCoefficients::Ptr m_plane_coefficients;
};
