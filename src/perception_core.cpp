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
    m_background_image_path = "/home/jiasen/det_ws/src/det_perception_core/image/background.png";
    m_foreground_image_mask = cv::Mat::zeros(720, 1280, CV_8UC1);
    m_background_image = cv::imread(m_background_image_path, cv::IMREAD_COLOR);
    m_lidar_topic = "/l515/depth_registered/points";
    m_pointcloud_sub = m_nh.subscribe(m_lidar_topic, 1, &PerceptionCore::pointcloudCallback, this);
    m_image_sub = m_nh.subscribe("/l515/color/image_raw", 1, &PerceptionCore::imageCallback, this);
    m_cropped_cloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/l515/points/cropped", 1);
    m_processed_cloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/l515/points/processed", 1);
    m_processed_image_pub = m_nh.advertise<sensor_msgs::Image>("/l515/image/processed", 1);
    m_foreground_image_pub = m_nh.advertise<sensor_msgs::Image>("/l515/image/foreground", 1);
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
    auto ordered_filtered_cloud = removePlane<pcl::PointXYZRGB>(ordered_cropped_cloud, m_foreground_image_mask, 
    m_plane_coefficients, 0.01);
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*ordered_filtered_cloud->cloud, cloud_msg);
    cloud_msg.header.frame_id = msg->header.frame_id;
    cloud_msg.header.stamp = msg->header.stamp;
    m_processed_cloud_pub.publish(cloud_msg);
    // convert ordered cloud to image
    auto foreground_rgb_image = orderedCloudToImage<pcl::PointXYZRGB>(ordered_filtered_cloud, 1280, 720);
    cv_bridge::CvImage foreground_rgb_msg;
    foreground_rgb_msg.header.stamp = msg->header.stamp;
    foreground_rgb_msg.header.frame_id = msg->header.frame_id;
    foreground_rgb_msg.encoding = sensor_msgs::image_encodings::BGR8;
    foreground_rgb_msg.image = foreground_rgb_image;
    m_foreground_image_pub.publish(foreground_rgb_msg.toImageMsg());


    // // convert ordered cloud to mask
    // auto foreground_mask = orderedCloudToMask<pcl::PointXYZRGB>(ordered_filtered_cloud, 1280, 720);
    // cv_bridge::CvImage foreground_mask_msg;
    // foreground_mask_msg.header.stamp = msg->header.stamp;
    // foreground_mask_msg.header.frame_id = msg->header.frame_id;
    // foreground_mask_msg.encoding = sensor_msgs::image_encodings::MONO8;
    // foreground_mask_msg.image = foreground_mask;
    // m_foreground_image_pub.publish(foreground_mask_msg.toImageMsg());


    // // connect components
    // cv::Mat labels;
    // int num_labels;
    // imageCluster(foreground_mask, labels, num_labels, 3000);
    // // log the number of labels
    // ROS_INFO_STREAM("Number of labels: " << num_labels);
    // // convert labels to image
    // cv::Mat labels_image;
    // labels.convertTo(labels_image, CV_8UC1);
    // // magnify the label values
    // labels_image *= 255 / num_labels;
    // cv::applyColorMap(labels_image, labels_image, cv::COLORMAP_JET);
    // cv_bridge::CvImage labels_image_msg;
    // labels_image_msg.header.stamp = msg->header.stamp;
    // labels_image_msg.header.frame_id = msg->header.frame_id;
    // labels_image_msg.encoding = sensor_msgs::image_encodings::BGR8;
    // labels_image_msg.image = labels_image;
    // m_foreground_image_pub.publish(labels_image_msg.toImageMsg());

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
const cv::Mat& foreground_mask, const pcl::ModelCoefficients::Ptr coefficients, const double& distance_threshold) {
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
            if (distance < distance_threshold && foreground_mask.at<uchar>(i + start_y, j + start_x) == 0) {
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

int main(int argc, char** argv)
{
    ros::init(argc, argv, "perception_core");
    ros::NodeHandle nh;
    PerceptionCore perception_core(nh);
    perception_core.run();
    ros::spin();
    return 0;
}
