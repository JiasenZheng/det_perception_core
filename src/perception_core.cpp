/**
 * @file perception_core.cpp
 * @author Jiasen Zheng
 * @brief Core class for perception tasks
 */
 
#include "perception_core.hpp"

PerceptionCore::PerceptionCore(ros::NodeHandle nh): m_nh(nh)
{
    m_nh.param("margin_pixels", m_margin_pixels, 20);
    m_image_count = 100;
    // init list of colors
    m_colors.push_back(cv::Vec3b(255, 0, 0));
    m_colors.push_back(cv::Vec3b(0, 255, 0));
    m_colors.push_back(cv::Vec3b(0, 0, 255));
    m_colors.push_back(cv::Vec3b(255, 255, 0));
    m_colors.push_back(cv::Vec3b(255, 0, 255));
    m_colors.push_back(cv::Vec3b(0, 255, 255));
    m_background_image_path = "/home/jiasen/det_ws/src/det_perception_core/image/background.png";
    m_foreground_image_mask = cv::Mat::zeros(720, 1280, CV_8UC1);
    m_foreground_cloud_mask = cv::Mat::zeros(720, 1280, CV_8UC1);
    m_background_image = cv::imread(m_background_image_path, cv::IMREAD_COLOR);
    m_lidar_topic = "/l515/depth_registered/points";
    m_pointcloud_sub = m_nh.subscribe(m_lidar_topic, 1, &PerceptionCore::pointcloudCallback, this);
    m_rgb_image_sub = m_nh.subscribe("/l515/color/image_raw", 1, &PerceptionCore::imageCallback, this);
    m_cropped_cloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/l515/points/cropped", 1);
    m_processed_cloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/l515/points/processed", 1);
    m_cluster_cloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/l515/points/cluster", 1);
    m_processed_image_pub = m_nh.advertise<sensor_msgs::Image>("/l515/image/processed", 1);
    m_foreground_image_pub = m_nh.advertise<sensor_msgs::Image>("/l515/image/foreground", 1);
    m_depth_image_pub = m_nh.advertise<sensor_msgs::Image>("/l515/image/depth", 1);
    m_processed_depth_image_pub = m_nh.advertise<sensor_msgs::Image>("/l515/image/depth/processed", 1);
}

void PerceptionCore::run()
{
}

void PerceptionCore::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{   
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *cloud);
    // convert the point cloud to ordered point cloud ptr
    typename OrderedCloud<pcl::PointXYZRGB>::Ptr ordered_raw_cloud(new OrderedCloud<pcl::PointXYZRGB>(cloud, 0, 0));
    // crop the point cloud
    auto ordered_cropped_cloud = cropOrderedCloud<pcl::PointXYZRGB>(ordered_raw_cloud, 0, 20, 40, 40);
    sensor_msgs::PointCloud2 cropped_msg;
    auto cropped_cloud = ordered_cropped_cloud->cloud;
    pcl::toROSMsg(*cropped_cloud, cropped_msg);
    cropped_msg.header.frame_id = msg->header.frame_id;
    cropped_msg.header.stamp = msg->header.stamp;
    m_cropped_cloud_pub.publish(cropped_msg);

    // remove the table plane
    if (m_plane_coefficients == nullptr)
    {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        planeSegmentation<pcl::PointXYZRGB>(ordered_cropped_cloud->cloud, 500, 0.01, inliers, coefficients);
        ROS_INFO_STREAM("Table detected!");
        m_plane_coefficients = coefficients;
        // log inliers and coefficients
        ROS_INFO_STREAM("Inliers: " << inliers->indices.size());
        ROS_INFO_STREAM("Coefficients: " << coefficients->values[0] << " " << coefficients->values[1] << " " <<
        coefficients->values[2] << " " << coefficients->values[3]);
        m_plane_coefficients = coefficients;
        return;
    }
    auto ordered_filtered_cloud = removePlane<pcl::PointXYZRGB>(ordered_cropped_cloud, m_plane_coefficients, 0.01);

    // convert ordered cloud to mask
    m_foreground_cloud_mask = orderedCloudToMask<pcl::PointXYZRGB>(ordered_filtered_cloud, 1280, 720);
    cv_bridge::CvImage foreground_mask_msg;
    foreground_mask_msg.header.stamp = msg->header.stamp;
    foreground_mask_msg.header.frame_id = msg->header.frame_id;
    foreground_mask_msg.encoding = sensor_msgs::image_encodings::MONO8;
    foreground_mask_msg.image = m_foreground_cloud_mask;
    m_foreground_image_pub.publish(foreground_mask_msg.toImageMsg());

    // remove noise in the mask
    auto foreground_mask = denoiseMask(m_foreground_cloud_mask, 11); 
    cv_bridge::CvImage foreground_mask_msg2;
    foreground_mask_msg2.header.stamp = msg->header.stamp;
    foreground_mask_msg2.header.frame_id = msg->header.frame_id;
    foreground_mask_msg2.encoding = sensor_msgs::image_encodings::MONO8;
    foreground_mask_msg2.image = foreground_mask;
    m_processed_depth_image_pub.publish(foreground_mask_msg2.toImageMsg());

    // mask the ordered cloud
    auto ordered_masked_cloud = maskOrderedCloud<pcl::PointXYZRGB>(ordered_filtered_cloud, foreground_mask);
    sensor_msgs::PointCloud2 masked_cloud_msg;
    pcl::toROSMsg(*ordered_masked_cloud->cloud, masked_cloud_msg);
    masked_cloud_msg.header.frame_id = msg->header.frame_id;
    masked_cloud_msg.header.stamp = msg->header.stamp;
    m_processed_cloud_pub.publish(masked_cloud_msg);

    // connected component analysis
    imageCluster(foreground_mask, m_image_labels, m_num_labels, m_bboxes);

    // draw bounding boxes
    auto bounding_boxes_image = drawBboxes(foreground_mask, m_bboxes);

    // downsample foreground mask
    auto downsampled_mask = downsampleMask(foreground_mask, 20);

    // get cluster clouds
    auto cluster_clouds = getClusterClouds(ordered_masked_cloud, m_image_labels, m_num_labels, downsampled_mask);

    // colorize the cluster clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clustered_cloud = colorizeClusters<pcl::PointXYZ>(cluster_clouds,
    m_colors);
    sensor_msgs::PointCloud2 colored_clustered_cloud_msg;
    pcl::toROSMsg(*colored_clustered_cloud, colored_clustered_cloud_msg);
    colored_clustered_cloud_msg.header.frame_id = msg->header.frame_id;
    colored_clustered_cloud_msg.header.stamp = msg->header.stamp;
    m_cluster_cloud_pub.publish(colored_clustered_cloud_msg);

    // get cluster oriented bounding boxes
    for (size_t i = 0; i < cluster_clouds.size(); i++)
    {
        auto cluster_cloud = cluster_clouds[i];
        Eigen::Vector3f position;
        Eigen::Quaternionf orientation;
        Eigen::Vector3f dimensions;
        computeOBB<pcl::PointXYZ>(cluster_cloud, position, orientation, dimensions);
        // publish a tf
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(position[0], position[1], position[2]));
        tf::Quaternion q(orientation.x(), orientation.y(), orientation.z(), orientation.w());
        transform.setRotation(q);
        m_br.sendTransform(tf::StampedTransform(transform, msg->header.stamp, msg->header.frame_id, "cluster_" + std::to_string(i)));
    }

}

void PerceptionCore::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    // convert to cv::Mat
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // get the image
    cv::Mat image = cv_ptr->image;

    // get foreground mask
    m_foreground_image_mask = imageBackgroundSubtraction(image, m_background_image, 60);

    // publish the processed image
    cv_bridge::CvImage out_msg;
    out_msg.header = msg->header;
    out_msg.encoding = sensor_msgs::image_encodings::MONO8;
    out_msg.image = m_foreground_image_mask;
    m_processed_image_pub.publish(out_msg.toImageMsg());
}

// void PerceptionCore::depthImageCallback(const sensor_msgs::ImageConstPtr& msg) {
//     // convert to cv::Mat
//     cv_bridge::CvImagePtr cv_ptr;
//     try
//     {
//         cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//         ROS_ERROR("cv_bridge exception: %s", e.what());
//         return;
//     }
//     // get the image
//     cv::Mat image = cv_ptr->image;

//     // auto contour_image = detectContour(image);
//     // cv_bridge::CvImage contour_msg;
//     // contour_msg.header.stamp = msg->header.stamp;
//     // contour_msg.header.frame_id = msg->header.frame_id;
//     // contour_msg.encoding = sensor_msgs::image_encodings::MONO8;
//     // contour_msg.image = contour_image;
//     // m_depth_image_pub.publish(contour_msg.toImageMsg());

//     // // mask depth image
//     // auto masked_image = maskDepthImage(image, m_foreground_cloud_mask);
//     // cv_bridge::CvImage masked_msg;
//     // masked_msg.header.stamp = msg->header.stamp;
//     // masked_msg.header.frame_id = msg->header.frame_id;
//     // masked_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
//     // masked_msg.image = masked_image;
//     // m_depth_image_pub.publish(masked_msg.toImageMsg());

//     // // outlier removal
//     // auto filtered_image = outlierRemoval(image, m_foreground_cloud_mask, 0.02);
//     // cv_bridge::CvImage filtered_msg;
//     // filtered_msg.header.stamp = msg->header.stamp;
//     // filtered_msg.header.frame_id = msg->header.frame_id;
//     // filtered_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
//     // filtered_msg.image = filtered_image;
//     // m_processed_depth_image_pub.publish(filtered_msg.toImageMsg());

// }

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
typename pcl::PointCloud<T>::Ptr PerceptionCore::cropOrderedCloud(const typename pcl::PointCloud<T>::Ptr cloud, 
const int& margin_pixels) {
    // loop through the ordered point cloud and remove points that are too close to the edge of the image
    typename pcl::PointCloud<T>::Ptr cloud_filtered(new pcl::PointCloud<T>);
    int width = cloud->width;
    int height = cloud->height;
    int new_width = width - 2 * margin_pixels;
    int new_height = height - 2 * margin_pixels;
    cloud_filtered->width = new_width;
    cloud_filtered->height = new_height;
    cloud_filtered->is_dense = false;
    cloud_filtered->points.resize(new_width * new_height);
    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            cloud_filtered->points[i * new_width + j] = cloud->points[(i + margin_pixels) * width + j + 
            margin_pixels];
        }
    }
    return cloud_filtered;
}

template <typename T>
typename pcl::PointCloud<T>::Ptr PerceptionCore::cropOrderedCloud(const typename pcl::PointCloud<T>::Ptr cloud,
const int& left_pixels, const int& right_pixels, const int& top_pixels, const int& bottom_pixels) {
    // loop through the ordered point cloud and remove points that are too close to the edge of the image
    typename pcl::PointCloud<T>::Ptr cloud_filtered(new pcl::PointCloud<T>);
    int width = cloud->width;
    int height = cloud->height;
    int new_width = width - left_pixels - right_pixels;
    int new_height = height - top_pixels - bottom_pixels;
    cloud_filtered->width = new_width;
    cloud_filtered->height = new_height;
    cloud_filtered->is_dense = false;
    cloud_filtered->points.resize(new_width * new_height);
    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            cloud_filtered->points[i * new_width + j] = cloud->points[(i + top_pixels) * width + j + left_pixels];
        }
    }
    return cloud_filtered;
}

template <typename T>
typename OrderedCloud<T>::Ptr PerceptionCore::cropOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
const int& margin_pixels) {
    // get ordered_cloud information
    int start_x = ordered_cloud->start_x + margin_pixels;
    int start_y = ordered_cloud->start_y + margin_pixels;
    // create a new ordered cloud
    typename OrderedCloud<T>::Ptr ordered_cloud_filtered(new OrderedCloud<T>);
    ordered_cloud_filtered->start_x = start_x;
    ordered_cloud_filtered->start_y = start_y;
    ordered_cloud_filtered->cloud = PerceptionCore::cropOrderedCloud<T>(ordered_cloud->cloud, margin_pixels);
    return ordered_cloud_filtered;
}

template <typename T>
typename OrderedCloud<T>::Ptr PerceptionCore::cropOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
const int& left_pixels, const int& right_pixels, const int& top_pixels, const int& bottom_pixels) {
    // get ordered_cloud information
    int start_x = ordered_cloud->start_x + left_pixels;
    int start_y = ordered_cloud->start_y + top_pixels;
    // create a new ordered cloud
    typename OrderedCloud<T>::Ptr ordered_cloud_filtered(new OrderedCloud<T>);
    ordered_cloud_filtered->start_x = start_x;
    ordered_cloud_filtered->start_y = start_y;
    ordered_cloud_filtered->cloud = PerceptionCore::cropOrderedCloud<T>(ordered_cloud->cloud, left_pixels, 
    right_pixels, top_pixels, bottom_pixels);
    return ordered_cloud_filtered;
}

cv::Mat PerceptionCore::imageBackgroundSubtraction(const cv::Mat& image, const cv::Mat& background, 
const int& threshold) {
    // background subtraction
    cv::Mat foreground_mask;
    cv::Mat background_image;
    cv::resize(background, background_image, image.size());
    cv::absdiff(image, background_image, foreground_mask);
    cv::cvtColor(foreground_mask, foreground_mask, cv::COLOR_BGR2GRAY);
    cv::threshold(foreground_mask, foreground_mask, threshold, 255, cv::THRESH_BINARY);

    return foreground_mask;
}

template <typename T>
typename OrderedCloud<T>::Ptr PerceptionCore::removePlane(const typename OrderedCloud<T>::Ptr ordered_cloud,
const pcl::ModelCoefficients::Ptr coefficients, const double& distance_threshold) {
    // get ordered_cloud information
    int start_x = ordered_cloud->start_x;
    int start_y = ordered_cloud->start_y;
    auto cloud = ordered_cloud->cloud;
    // create a new ordered cloud
    typename OrderedCloud<T>::Ptr ordered_cloud_filtered(new OrderedCloud<T>);
    ordered_cloud_filtered->start_x = start_x;
    ordered_cloud_filtered->start_y = start_y;
    typename pcl::PointCloud<T>::Ptr cloud_filtered(new pcl::PointCloud<T>);
    int width = cloud->width;
    int height = cloud->height;
    cloud_filtered->width = width;
    cloud_filtered->height = height;
    cloud_filtered->is_dense = false;
    cloud_filtered->points.resize(cloud->points.size());
    double threshold_overhead = 0.02;
    auto max_z = std::abs(coefficients->values[3]);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            double distance = std::abs(coefficients->values[0] * cloud->points[idx].x + coefficients->values[1] *
            cloud->points[idx].y + coefficients->values[2] * cloud->points[idx].z + coefficients->values[3]) /
            std::sqrt(coefficients->values[0] * coefficients->values[0] + coefficients->values[1] *
            coefficients->values[1] + coefficients->values[2] * coefficients->values[2]);
            // set the point to be invalid if it is below the plane
            if (cloud->points[idx].z > (max_z + threshold_overhead)) {
                cloud_filtered->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                cloud_filtered->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                cloud_filtered->points[idx].z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            // set the point to be invalid if it is too close to the plane and the foreground mask is 0
            // if (distance < distance_threshold && m_foreground_image_mask.at<uchar>(i + start_y, j + start_x) == 0) {
            if (distance < distance_threshold) {
                cloud_filtered->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                cloud_filtered->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                cloud_filtered->points[idx].z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            cloud_filtered->points[idx] = cloud->points[idx];
        }
    }
    ordered_cloud_filtered->cloud = cloud_filtered;
    // shrink the ordered cloud
    ordered_cloud_filtered = PerceptionCore::shrinkOrderedCloud<T>(ordered_cloud_filtered);
    return ordered_cloud_filtered;
}

template <typename T>
typename OrderedCloud<T>::Ptr PerceptionCore::shrinkOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud) {
    // shrink the ordered point cloud by removing the invalid points
    typename OrderedCloud<T>::Ptr cloud_filtered(new OrderedCloud<T>);
    int start_x = ordered_cloud->start_x;
    int start_y = ordered_cloud->start_y;
    int width = ordered_cloud->cloud->width;
    int height = ordered_cloud->cloud->height;
    int pointer_u = 0;
    int pointer_d = height - 1;
    int pointer_l = 0;
    int pointer_r = width - 1;
    // shrink the point cloud from the top and bottom using two pointers
    while (pointer_u < pointer_d) {
        bool flag_u = false;
        bool flag_d = false;
        for (int i = 0; i < width; i++) {
            if (!std::isnan(ordered_cloud->cloud->points[pointer_u * width + i].x)) {
                flag_u = true;
                break;
            }
        }
        for (int i = 0; i < width; i++) {
            if (!std::isnan(ordered_cloud->cloud->points[pointer_d * width + i].x)) {
                flag_d = true;
                break;
            }
        }
        if (flag_u && flag_d) {
            break;
        }
        if (!flag_u) {
            pointer_u++;
            start_y++;
        }
        if (!flag_d) {
            pointer_d--;
        }
    }
    // get the new height
    int new_height = pointer_d - pointer_u + 1;
    // shrink the point cloud from the left and right using two pointers
    while (pointer_l < pointer_r) {
        bool flag_l = false;
        bool flag_r = false;
        for (int i = 0; i < new_height; i++) {
            if (!std::isnan(ordered_cloud->cloud->points[(i + pointer_u) * width + pointer_l].x)) {
                flag_l = true;
                break;
            }
        }
        for (int i = 0; i < new_height; i++) {
            if (!std::isnan(ordered_cloud->cloud->points[(i + pointer_u) * width + pointer_r].x)) {
                flag_r = true;
                break;
            }
        }
        if (flag_l && flag_r) {
            break;
        }
        if (!flag_l) {
            pointer_l++;
            start_x++;
        }
        if (!flag_r) {
            pointer_r--;
        }
    }
    // get the new width
    int new_width = pointer_r - pointer_l + 1;
    // set the new point cloud
    cloud_filtered->cloud->width = new_width;
    cloud_filtered->cloud->height = new_height;
    cloud_filtered->cloud->is_dense = false;
    cloud_filtered->cloud->points.resize(new_width * new_height);
    cloud_filtered->start_x = start_x;
    cloud_filtered->start_y = start_y;
    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            cloud_filtered->cloud->points[i * new_width + j] = ordered_cloud->cloud->points[(i + pointer_u) * width +
            j + pointer_l];
        }
    }
    return cloud_filtered;
}

template <typename T>
void PerceptionCore::getPlaneLimits(const typename pcl::PointCloud<T>::Ptr cloud, const pcl::PointIndices::Ptr inliers, 
std::vector<double>& limits) {
    // get the plane limits
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::min();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::min();
    for (int i = 0; i < inliers->indices.size(); i++) {
        int idx = inliers->indices[i];
        if (cloud->points[idx].x < min_x) {
            min_x = cloud->points[idx].x;
        }
        if (cloud->points[idx].x > max_x) {
            max_x = cloud->points[idx].x;
        }
        if (cloud->points[idx].y < min_y) {
            min_y = cloud->points[idx].y;
        }
        if (cloud->points[idx].y > max_y) {
            max_y = cloud->points[idx].y;
        }
        if (cloud->points[idx].z < min_z) {
            min_z = cloud->points[idx].z;
        }
        if (cloud->points[idx].z > max_z) {
            max_z = cloud->points[idx].z;
        }
    }
    limits.push_back(min_x);
    limits.push_back(max_x);
    limits.push_back(min_y);
    limits.push_back(max_y);
    limits.push_back(min_z);
    limits.push_back(max_z);
}

template <typename T>
cv::Mat PerceptionCore::orderedCloudToImage(const typename OrderedCloud<T>::Ptr ordered_cloud, const int& width, 
const int& height) {
    // convert the ordered point cloud to an image
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < ordered_cloud->cloud->height; i++) {
        for (size_t j = 0; j < ordered_cloud->cloud->width; j++) {
            int x = ordered_cloud->start_x + j;
            int y = ordered_cloud->start_y + i;
            if (!std::isnan(ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].x)) {
                image.at<cv::Vec3b>(y, x)[0] = ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].b;
                image.at<cv::Vec3b>(y, x)[1] = ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].g;
                image.at<cv::Vec3b>(y, x)[2] = ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].r;
            }
        }
    }
    return image;
}

template <typename T>
cv::Mat PerceptionCore::orderedCloudToMask(const typename OrderedCloud<T>::Ptr ordered_cloud, const int& width,
const int& height) {
    // convert the ordered point cloud to a mask
    cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < ordered_cloud->cloud->height; i++) {
        for (size_t j = 0; j < ordered_cloud->cloud->width; j++) {
            int x = ordered_cloud->start_x + j;
            int y = ordered_cloud->start_y + i;
            // if the point is valid, set the mask to 255
            if (!std::isnan(ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].x)) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }
    return mask;
}

void PerceptionCore::imageCluster(const cv::Mat& mask, cv::Mat& labels, int& num_labels, const int& pixel_threshold) {
    // connect the components in the mask
    cv::Mat mask_8u;
    mask.convertTo(mask_8u, CV_8U);
    num_labels = cv::connectedComponents(mask_8u, labels, 8, CV_32S);
    // remove components with less than 100 pixels
    for (int i = 0; i < num_labels; i++) {
        if (cv::countNonZero(labels == i) < pixel_threshold) {
            labels.setTo(0, labels == i);
            num_labels--;
        }
    }
    // find the largest component and the smallest component
    int max_pixels = 0;
    int min_pixels = std::numeric_limits<int>::max();
    for (int i = 0; i < num_labels; i++) {
        int pixels = cv::countNonZero(labels == i);
        if (pixels > max_pixels) {
            max_pixels = pixels;
        }
        if (pixels < min_pixels && pixels > 0) {
            min_pixels = pixels;
        }
    }
    // log the number of pixels in the largest component and the smallest component
    std::cout << "max pixels: " << max_pixels << std::endl;
    std::cout << "min pixels: " << min_pixels << std::endl;
}

void PerceptionCore::imageCluster(const cv::Mat& mask, cv::Mat& labels, int& num_labels, 
std::vector<cv::Rect>& bboxes) {
    // find the connected components
    cv::Mat mask_8u;
    mask.convertTo(mask_8u, CV_8U);
    num_labels = cv::connectedComponents(mask_8u, labels);
    num_labels--;
    // compute bounding boxes for each component
    bboxes.resize(num_labels);
    for (int i = 0; i < num_labels; i++) {
        bboxes[i] = cv::Rect(0, 0, 0, 0);
    }
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (labels.at<int>(i, j) == 0) {
                continue;
            }
            int label = labels.at<int>(i, j) - 1;
            if (bboxes[label].x == 0) {
                bboxes[label].x = j;
            }
            else if (bboxes[label].x > j) {
                bboxes[label].width += bboxes[label].x - j;
                bboxes[label].x = j;
            }
            if (bboxes[label].y == 0) {
                bboxes[label].y = i;
            }
            else if (bboxes[label].y > i) {
                bboxes[label].height += bboxes[label].y - i;
                bboxes[label].y = i;
            }
            if (bboxes[label].x + bboxes[label].width < j) {
                bboxes[label].width = j - bboxes[label].x;
            }
            if (bboxes[label].y + bboxes[label].height < i) {
                bboxes[label].height = i - bboxes[label].y;
            }
        }
    }
}

cv::Mat PerceptionCore::drawBboxes(const cv::Mat& image, const std::vector<cv::Rect>& bboxes) {
    // convert the image to RGB
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_GRAY2RGB);
    // draw the bounding boxes
    for (size_t i = 0; i < bboxes.size(); i++) {
        cv::rectangle(image_rgb, bboxes[i], cv::Scalar(0, 255, 0), 2);
    }
    return image_rgb;
}

cv::Mat PerceptionCore::detectContour(const cv::Mat& depth_image) {
    // convert image to cv_8u
    cv::Mat depth_image_8u;
    depth_image.convertTo(depth_image_8u, CV_8U);
    // detect the contour
    cv::Mat contour;
    cv::Canny(depth_image_8u, contour, 100, 200);
    return contour;
}

cv::Mat PerceptionCore::maskDepthImage(const cv::Mat& depth_image, const cv::Mat& mask) {
    // mask the depth image
    cv::Mat masked_depth_image;
    depth_image.copyTo(masked_depth_image, mask);
    return masked_depth_image;
}

cv::Mat PerceptionCore::outlierRemoval(const cv::Mat& depth_image, const cv::Mat& mask, const int& threshold) {
    cv::Mat masked_depth_image;
    depth_image.copyTo(masked_depth_image, mask);
    // remove the outliers
    int num_rows = masked_depth_image.rows;
    int num_cols = masked_depth_image.cols;
    for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
        if (masked_depth_image.at<uchar>(i, j) == 0) {
            continue;
        }
        // int neighbor_count = 0;
        int outlier_count = 0;
        for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
            if (i + k < 0 || i + k >= num_rows || j + l < 0 || j + l >= num_cols) {
                continue;
            }
            if (masked_depth_image.at<uchar>(i + k, j + l) == 0) {
                continue;
            }
            // neighbor_count++;
            if (std::abs(masked_depth_image.at<uchar>(i, j) - masked_depth_image.at<uchar>(i + k, j + l)) > threshold) {
                outlier_count++;
            }
        }
        // // remove if have too few neighbors
        // if (neighbor_count < 4) {
        //     masked_depth_image.at<uchar>(i, j) = 0;
        // }
        // remove if too many outliers
        if (outlier_count > 2) {
            masked_depth_image.at<uchar>(i, j) = 0;
        }
        }
    }
    }
    return masked_depth_image;
}

template <typename T>
typename OrderedCloud<T>::Ptr PerceptionCore::maskOrderedCloud(const typename OrderedCloud<T>::Ptr ordered_cloud,
const cv::Mat& mask) {
    // mask the ordered point cloud
    typename OrderedCloud<T>::Ptr masked_ordered_cloud(new OrderedCloud<T>);
    masked_ordered_cloud->cloud = typename pcl::PointCloud<T>::Ptr(new pcl::PointCloud<T>);
    masked_ordered_cloud->start_x = ordered_cloud->start_x;
    masked_ordered_cloud->start_y = ordered_cloud->start_y;
    masked_ordered_cloud->cloud->width = ordered_cloud->cloud->width;
    masked_ordered_cloud->cloud->height = ordered_cloud->cloud->height;
    masked_ordered_cloud->cloud->is_dense = false;
    masked_ordered_cloud->cloud->points.resize(masked_ordered_cloud->cloud->width * 
    masked_ordered_cloud->cloud->height);
    for (size_t i = 0; i < ordered_cloud->cloud->height; i++) {
        for (size_t j = 0; j < ordered_cloud->cloud->width; j++) {
            int x = ordered_cloud->start_x + j;
            int y = ordered_cloud->start_y + i;
            if ((mask.at<uchar>(y, x) == 255) && 
            (!std::isnan(ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].x))) {
                masked_ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j] = 
                ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j];
            }
            else {
                masked_ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].x = 
                std::numeric_limits<float>::quiet_NaN();
                masked_ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].y = 
                std::numeric_limits<float>::quiet_NaN();
                masked_ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].z = 
                std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    return masked_ordered_cloud;
}

cv::Mat PerceptionCore::denoiseMask(const cv::Mat& mask, const int& kernel_size) {
    // remove noise in the mask
    cv::Mat denoised_mask = mask.clone();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(denoised_mask, denoised_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(denoised_mask, denoised_mask, cv::MORPH_CLOSE, kernel);
    return denoised_mask;
}

void PerceptionCore::imageCluster(const cv::Mat& mask, cv::Mat& labels, int& num_labels) {
    // find the connected components
    cv::Mat mask_8u;
    mask.convertTo(mask_8u, CV_8U);
    num_labels = cv::connectedComponents(mask_8u, labels);
    num_labels--;
}

template <typename T>
std::vector<typename pcl::PointCloud<T>::Ptr> PerceptionCore::getClusterClouds(
const typename OrderedCloud<T>::Ptr ordered_cloud,
const cv::Mat& labels, const int& num_labels) {
    // get the point clouds of each cluster
    std::vector<typename pcl::PointCloud<T>::Ptr> cluster_clouds(num_labels);
    for (int i = 0; i < num_labels; i++) {
        cluster_clouds[i] = typename pcl::PointCloud<T>::Ptr(new pcl::PointCloud<T>);
    }
    for (size_t i = 0; i < ordered_cloud->cloud->height; i++) {
    for (size_t j = 0; j < ordered_cloud->cloud->width; j++) {
        int x = ordered_cloud->start_x + j;
        int y = ordered_cloud->start_y + i;
        if (labels.at<int>(y, x) == 0) {
            continue;
        }
        // check if the point is valid
        if (std::isnan(ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].x)) {
            continue;
        }
        cluster_clouds[labels.at<int>(y, x) - 1]->points.push_back(ordered_cloud->cloud->points[i * 
        ordered_cloud->cloud->width + j]);
    }
    }
    for (int i = 0; i < num_labels; i++) {
        cluster_clouds[i]->width = cluster_clouds[i]->points.size();
        cluster_clouds[i]->height = 1;
        cluster_clouds[i]->is_dense = true;
    }
    return cluster_clouds;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> PerceptionCore::getClusterClouds(
const typename OrderedCloud<pcl::PointXYZRGB>::Ptr ordered_cloud,
const cv::Mat& labels, const int& num_labels, const cv::Mat& downsampled_mask) {
    // get the point clouds of each cluster
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_clouds(num_labels);
    for (int i = 0; i < num_labels; i++) {
        cluster_clouds[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    }
    for (size_t i = 0; i < ordered_cloud->cloud->height; i++) {
    for (size_t j = 0; j < ordered_cloud->cloud->width; j++) {
        int x = ordered_cloud->start_x + j;
        int y = ordered_cloud->start_y + i;
        if (labels.at<int>(y, x) == 0) {
            continue;
        }
        if (downsampled_mask.at<uchar>(y, x) == 0) {
            continue;
        }
        // check if the point is valid
        if (std::isnan(ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].x)) {
            continue;
        }
        pcl::PointXYZ point;
        point.x = ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].x;
        point.y = ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].y;
        point.z = ordered_cloud->cloud->points[i * ordered_cloud->cloud->width + j].z;
        cluster_clouds[labels.at<int>(y, x) - 1]->points.push_back(point);
    }
    }
    for (int i = 0; i < num_labels; i++) {
        cluster_clouds[i]->width = cluster_clouds[i]->points.size();
        cluster_clouds[i]->height = 1;
        cluster_clouds[i]->is_dense = true;
    }
    return cluster_clouds;
}

template <typename T>
pcl::PointCloud<pcl::PointXYZRGB>::Ptr PerceptionCore::colorizeClusters(
std::vector<typename pcl::PointCloud<T>::Ptr> cluster_clouds,
std::vector<cv::Vec3b> colors) {
    // colorize the clusters
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    auto num_colors = colors.size();
    for (size_t i = 0; i < cluster_clouds.size(); i++) {
        for (size_t j = 0; j < cluster_clouds[i]->points.size(); j++) {
            pcl::PointXYZRGB point;
            point.x = cluster_clouds[i]->points[j].x;
            point.y = cluster_clouds[i]->points[j].y;
            point.z = cluster_clouds[i]->points[j].z;
            point.r = colors[i % num_colors][0];
            point.g = colors[i % num_colors][1];
            point.b = colors[i % num_colors][2];
            colored_cloud->points.push_back(point);
        }
    }
    colored_cloud->width = colored_cloud->points.size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = true;
    return colored_cloud;
}

template <typename T>
void PerceptionCore::computeOBB(const typename pcl::PointCloud<T>::Ptr cloud, Eigen::Vector3f& position,
Eigen::Quaternionf& orientation, Eigen::Vector3f& dimensions) {
    // compute the oriented bounding box
    pcl::MomentOfInertiaEstimation<T> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();
    T min_point_OBB;
    T max_point_OBB;
    T position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    position = position_OBB.getVector3fMap();
    orientation = Eigen::Quaternionf(rotational_matrix_OBB);
    dimensions = (max_point_OBB.getVector3fMap() - min_point_OBB.getVector3fMap()).cwiseAbs();
}

cv::Mat PerceptionCore::downsampleMask(const cv::Mat& mask, const int& factor) {
    // copy mask
    cv::Mat downsampled_mask = mask.clone();
    // randomly turn off pixels
    for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
        if (mask.at<uchar>(i, j) == 0) {
            continue;
        }
        if (rand() % factor != 0) {
            downsampled_mask.at<uchar>(i, j) = 0;
        }
    }
    }
    return downsampled_mask;
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
