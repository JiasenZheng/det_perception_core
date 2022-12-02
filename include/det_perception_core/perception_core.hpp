/**
 * @file perception_core.hpp
 * @author Jiasen Zheng
 * @brief Core class for perception tasks
 */

#pragma once

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
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
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/centroid.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include "det_perception_core/Inference.h"

// data structure to store ordered point cloud and its start pixel coordinate
template <typename T>
struct OrderedCloud
{
    // constructor
    OrderedCloud() : start_x(0), start_y(0)
    {
        // initialize point cloud
        cloud.reset(new pcl::PointCloud<T>);
    };
    OrderedCloud(typename pcl::PointCloud<T>::Ptr cloud, int x, int y) : cloud(cloud), start_x(x), start_y(y){};
    // data
    typename pcl::PointCloud<T>::Ptr cloud;
    int start_x;
    int start_y;
    // create a pointer
    typedef boost::shared_ptr<OrderedCloud<T>> Ptr;
};

struct Cluster
{
    std::vector<int> pixel_center;
    tf::Transform pose;
    double scale;
}

class PerceptionCore
{
public:
    PerceptionCore(ros::NodeHandle nh);
    void run();
    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);
    template <typename T>
    typename OrderedCloud<T>::Ptr shrinkOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud);
    template <typename T>
    void planeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud, int max_iterations,
                           double distance_threshold, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
    template <typename T>
    pcl::PointCloud<pcl::PointXYZRGB> colorizeSegmentation(const typename pcl::PointCloud<T>::Ptr cloud,
                                                           const pcl::PointIndices::Ptr inliers);
    template <typename T>
    typename pcl::PointCloud<T>::Ptr cropOrderedCloud(const typename pcl::PointCloud<T>::Ptr cloud,
                                                      const int &margin_pixels);
    template <typename T>
    typename OrderedCloud<T>::Ptr cropOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
                                                   const int &margin_pixels);
    template <typename T>
    typename pcl::PointCloud<T>::Ptr cropOrderedCloud(const typename pcl::PointCloud<T>::Ptr cloud,
                                                      const int &left_pixels, const int &right_pixels, const int &top_pixels, const int &bottom_pixels);
    template <typename T>
    typename OrderedCloud<T>::Ptr cropOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
                                                   const int &left_pixels, const int &right_pixels, const int &top_pixels, const int &bottom_pixels);
    cv::Mat imageBackgroundSubtraction(const cv::Mat &image, const cv::Mat &background, const int &threshold);
    template <typename T>
    typename OrderedCloud<T>::Ptr removePlane(const typename OrderedCloud<T>::Ptr ordered_cloud,
                                              const pcl::ModelCoefficients::Ptr coefficients, const double &distance_threshold);
    template <typename T>
    void getPlaneLimits(const typename pcl::PointCloud<T>::Ptr cloud, const pcl::PointIndices::Ptr inliers,
                        std::vector<double> &limits);
    template <typename T>
    cv::Mat orderedCloudToImage(const typename OrderedCloud<T>::Ptr ordered_cloud, const int &width, const int &height);
    template <typename T>
    cv::Mat orderedCloudToMask(const typename OrderedCloud<T>::Ptr ordered_cloud, const int &width, const int &height);
    void imageCluster(const cv::Mat &mask, cv::Mat &labels, int &num_labels, const int &pixel_threshold);
    cv::Mat detectContour(const cv::Mat &depth_image);
    cv::Mat maskDepthImage(const cv::Mat &depth_image, const cv::Mat &mask);
    cv::Mat outlierRemoval(const cv::Mat &depth_image, const cv::Mat &mask, const int &threshold);
    cv::Mat denoiseMask(const cv::Mat &mask, const int &kernel_size);
    template <typename T>
    typename OrderedCloud<T>::Ptr maskOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
                                                   const cv::Mat &mask);
    void imageCluster(const cv::Mat &mask, cv::Mat &labels, int &num_labels);
    void imageCluster(const cv::Mat &mask, cv::Mat &labels, int &num_labels, std::vector<cv::Rect> &bboxes);
    cv::Mat drawBboxes(const cv::Mat &image, const std::vector<cv::Rect> &bboxes);
    template <typename T>
    std::vector<typename pcl::PointCloud<T>::Ptr> getClusterClouds(const typename OrderedCloud<T>::Ptr ordered_cloud,
                                                                   const cv::Mat &labels, const int &num_labels);
    template <typename T>
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorizeClusters(
        std::vector<typename pcl::PointCloud<T>::Ptr> cluster_clouds,
        std::vector<cv::Vec3b> colors);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getClusterClouds(
        const typename OrderedCloud<pcl::PointXYZRGB>::Ptr ordered_cloud,
        const cv::Mat &labels, const int &num_labels, const cv::Mat &downsampled_mask);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getClusterClouds(
        const typename OrderedCloud<pcl::PointXYZRGB>::Ptr ordered_cloud,
        const cv::Mat &downsampled_mask, const int &num_inferences);
    template <typename T>
    void computeOBB(const typename pcl::PointCloud<T>::Ptr cloud, Eigen::Vector3f &position,
                    Eigen::Quaternionf &orientation, Eigen::Vector3f &dimensions);
    cv::Mat downsampleMask(const cv::Mat &mask, const int &factor);
    std::vector<cv::Rect> expandBoundingBoxes(const std::vector<cv::Rect> &bboxes, const int &pixels);
    cv::Mat mergeMasks(const cv::Mat &foreground_mask, const std::vector<unsigned char> &masks, const int &width,
                       const int &height, const int &num_labels);
    void loadMesh(const std::string &filename, pcl::PolygonMesh &mesh, Eigen::Matrix4f &transform,
                  Eigen::Vector3f &dimensions);
    double computeClusterDiff(const cv::Mat &prev_mask, const cv::Mat &curr_mask, const cv::Rect &bbox);
    bool clusterDiffStateMachine(const cv::Mat &prev_mask, const cv::Mat &curr_mask, 
    const std::vector<cv::Rect> &bboxes, const double &threshold, std::vector<bool> &diffs);
    cv::Mat updateForegroundMask(const cv::Mat &foreground_mask, const std::vector<bool> &diffs,
                                 const std::vector<cv::Rect> &bboxes);
    cv::Mat updateForegroundMask(const cv::Mat &foreground_mask, const std::vector<bool> &diffs,
                                 const cv::Mat &labels);
    cv::Mat splitMask(const cv::Mat &foreground_mask, const std::vector<bool> &diffs, const cv::Mat &labels);

private:
    int m_height;
    int m_width;
    int m_num_clusters_prev;
    ros::NodeHandle m_nh;
    Eigen::Vector3f m_dimensions;
    Eigen::Matrix4f m_transform;
    tf::TransformBroadcaster m_br;
    tf::Transform m_table_tf;
    std::vector<double> m_plane_limits;
    std::vector<cv::Vec3b> m_colors;
    std::vector<Cluster> m_clusters_prev;
    pcl::ModelCoefficients::Ptr m_plane_coefficients;
    cv::Mat m_raw_image;
    cv::Mat m_background_image;
    cv::Mat m_foreground_image_mask;
    cv::Mat m_foreground_cloud_mask;
    cv::Mat m_foreground_mask_prev;
    std::string m_background_image_path;
    std::string m_stl_mesh_path;
    std::string m_lidar_topic;
    ros::ServiceClient m_infer_client;
    ros::Subscriber m_pointcloud_sub;
    ros::Subscriber m_rgb_image_sub;
    ros::Publisher m_cropped_cloud_pub;
    ros::Publisher m_processed_cloud_pub;
    ros::Publisher m_cluster_cloud_pub;
    ros::Publisher m_processed_image_pub;
    ros::Publisher m_foreground_image_pub;
    ros::Publisher m_depth_image_pub;
    ros::Publisher m_processed_depth_image_pub;
    ros::Publisher m_marker_pub;
    det_perception_core::Inference m_infer_srv;
};
