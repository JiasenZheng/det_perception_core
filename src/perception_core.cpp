/**
 * @file perception_core.cpp
 * @author Jiasen Zheng
 * @brief Core class for perception tasks
 */
 
#include "perception_core.hpp"

PerceptionCore::PerceptionCore(ros::NodeHandle nh): m_nh(nh)
{
    m_raw_image = cv::Mat::zeros(480, 640, CV_8UC3);
    m_height = 720;
    m_width = 1280;
    m_num_clusters_prev = 0;
    // init list of colors
    m_colors.push_back(cv::Vec3b(255, 0, 0));
    m_colors.push_back(cv::Vec3b(0, 255, 0));
    m_colors.push_back(cv::Vec3b(0, 0, 255));
    m_colors.push_back(cv::Vec3b(255, 255, 0));
    m_colors.push_back(cv::Vec3b(255, 0, 255));
    m_colors.push_back(cv::Vec3b(0, 255, 255));
    // init m_clusters_prev as an empty vector
    m_clusters_prev = std::vector<Cluster>();
    m_background_image_path = "/home/jiasen/det_ws/src/det_perception_core/image/background.png";
    m_stl_mesh_path = "/home/jiasen/det_ws/src/det_perception_core/meshes/nontextured.stl";
    m_foreground_image_mask = cv::Mat::zeros(m_height, m_width, CV_8UC1);
    m_foreground_cloud_mask = cv::Mat::zeros(m_height, m_width, CV_8UC1);
    m_foreground_mask_prev = cv::Mat::zeros(m_height, m_width, CV_8UC1);
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
    m_marker_pub = m_nh.advertise<visualization_msgs::Marker>("/l515/marker", 0);
    // wait for service
    ros::service::waitForService("/infer");
    // create an infer service
    m_infer_client = m_nh.serviceClient<det_perception_core::Inference>("/infer");
    m_infer_srv.request.start_x = 0;
    m_infer_srv.request.start_y = 0;
    m_infer_srv.request.width = m_width;
    m_infer_srv.request.height = m_height;
    // load the mesh and compute the centroid of the mesh
    pcl::PolygonMesh mesh;
    loadMesh(m_stl_mesh_path, mesh, m_transform, m_dimensions);
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

    if (m_plane_coefficients == nullptr)
    {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        planeSegmentation<pcl::PointXYZRGB>(ordered_cropped_cloud->cloud, 500, 0.01, inliers, coefficients);
        ROS_INFO_STREAM("Table detected!");
        m_plane_coefficients = coefficients;
        // find the normal of the plane
        Eigen::Vector3f normal(m_plane_coefficients->values[0], m_plane_coefficients->values[1], 
        m_plane_coefficients->values[2]);
        // convert the normal to quaternion
        Eigen::Quaternionf q;
        q.setFromTwoVectors(Eigen::Vector3f::UnitZ(), normal);
        m_table_tf.setOrigin(tf::Vector3(0, 0, -m_plane_coefficients->values[3]));
        m_table_tf.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
        // find the x y limits of the plane 
        getPlaneLimits<pcl::PointXYZRGB>(ordered_cropped_cloud->cloud, inliers, m_plane_limits);

        //publish the table tf
        m_br.sendTransform(tf::StampedTransform(m_table_tf, msg->header.stamp, msg->header.frame_id, "table"));
        // publish the table marker
        visualization_msgs::Marker table_marker;
        table_marker.header.frame_id = msg->header.frame_id;
        table_marker.header.stamp = msg->header.stamp;
        table_marker.ns = "table";
        table_marker.id = 0;
        table_marker.type = visualization_msgs::Marker::CUBE;
        table_marker.action = visualization_msgs::Marker::ADD;
        table_marker.pose.position.x = m_table_tf.getOrigin().x();
        table_marker.pose.position.y = m_table_tf.getOrigin().y();
        table_marker.pose.position.z = m_table_tf.getOrigin().z();
        table_marker.pose.orientation.x = m_table_tf.getRotation().x();
        table_marker.pose.orientation.y = m_table_tf.getRotation().y();
        table_marker.pose.orientation.z = m_table_tf.getRotation().z();
        table_marker.pose.orientation.w = m_table_tf.getRotation().w();
        table_marker.scale.x = std::abs(m_plane_limits[0] - m_plane_limits[1]);
        table_marker.scale.y = std::abs(m_plane_limits[2] - m_plane_limits[3]);
        table_marker.scale.z = 0.02;
        table_marker.color.a = 1.0;
        table_marker.color.r = 0.44;
        table_marker.color.g = 0.33;
        table_marker.color.b = 0.23;
        m_marker_pub.publish(table_marker);

        return;
    }
    // remove the table plane
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

    // image cluster the mask
    cv::Mat labels;
    int num_labels;
    std::vector<cv::Rect> bboxes;
    imageCluster(foreground_mask, labels, num_labels, bboxes);

    // // state machine
    // std::vector<bool> diffs(num_labels, false);
    // bool need_inference = clusterDiffStateMachine(m_foreground_mask_prev, foreground_mask, bboxes, 0.15, diffs);
    // m_foreground_mask_prev = foreground_mask;

    // if (!need_inference) {
    //     return;
    // }

    // new state machine
    auto stationary_moving_areas = computeStationaryMovingAreas(m_foreground_mask_prev, foreground_mask, bboxes, 0.15);
    m_foreground_mask_prev = foreground_mask;

    // // publish stationary_moving_areas
    // cv_bridge::CvImage stationary_moving_areas_msg;
    // stationary_moving_areas_msg.header.stamp = msg->header.stamp;
    // stationary_moving_areas_msg.header.frame_id = msg->header.frame_id;
    // stationary_moving_areas_msg.encoding = sensor_msgs::image_encodings::MONO8;
    // stationary_moving_areas_msg.image = stationary_moving_areas;
    // m_depth_image_pub.publish(stationary_moving_areas_msg.toImageMsg());

    // make an inference
    int num_inferences = 0;
    std::vector<unsigned char> inference_masks;
    
    if (m_infer_client.call(m_infer_srv))
    {
        num_inferences = m_infer_srv.response.num_inferences;
        inference_masks = m_infer_srv.response.data;
        for (int i = 0; i < num_inferences; ++i) {
            // create a mask of bool
            cv::Mat mask = cv::Mat::zeros(m_height, m_width, CV_8UC1);
            for (int row = 0; row < m_height; ++row) {
                for (int col = 0; col < m_width; ++col) {
                    auto idx = row * m_width + col;
                    idx += i * m_width * m_height;
                    if (inference_masks[idx] == true) {
                        mask.at<uchar>(row, col) = 255;
                    }
                }
            }
        }
    }
    else
    {
        ROS_ERROR_STREAM("Failed to call inference service!");
    }

    // merge foreground mask and inference mask
    auto merged_mask = mergeMasks(foreground_mask, inference_masks, m_width, m_height, num_inferences);
    
    // compute the pixel centroid of the cluster
    auto centroids = computeClusterPixelCentroids(merged_mask, num_inferences);

    // downsample foreground mask
    auto downsampled_mask = downsampleMask(merged_mask, 20);

    // get cluster clouds
    auto cluster_clouds = getClusterClouds(ordered_filtered_cloud, downsampled_mask, num_inferences);

    // // colorize the cluster clouds
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clustered_cloud = colorizeClusters<pcl::PointXYZ>(cluster_clouds,
    // m_colors);
    // sensor_msgs::PointCloud2 colored_clustered_cloud_msg;
    // pcl::toROSMsg(*colored_clustered_cloud, colored_clustered_cloud_msg);
    // colored_clustered_cloud_msg.header.frame_id = msg->header.frame_id;
    // colored_clustered_cloud_msg.header.stamp = msg->header.stamp;
    // m_cluster_cloud_pub.publish(colored_clustered_cloud_msg);
    std::vector<Cluster> clusters_curr;
    // get cluster oriented bounding boxes
    for (size_t i = 0; i < centroids.size(); i++)
    {
        auto centroid = centroids[i];
        if (stationary_moving_areas.at<uchar>(centroid.first, centroid.second) == 127) {
            // find the associated cluster
            int closest_cluster_idx = -1;
            float min_dist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < m_clusters_prev.size(); j++) {
                auto cluster_prev = m_clusters_prev[j];
                auto centroid_prev = cluster_prev.pixel_center;
                float dist = std::sqrt(std::pow(centroid.first - centroid_prev.first, 2) + std::pow(centroid.second - centroid_prev.second, 2));
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_cluster_idx = j;
                }
            }
            auto associated_cluster = m_clusters_prev[closest_cluster_idx];
            tf::Transform pose = associated_cluster.pose;
            auto scale = associated_cluster.scale;
            visualization_msgs::Marker marker;
            marker.header.frame_id = msg->header.frame_id;
            marker.header.stamp = msg->header.stamp;
            marker.ns = "cluster_" + std::to_string(i);
            marker.id = 0;
            marker.type = visualization_msgs::Marker::MESH_RESOURCE;
            marker.mesh_resource = "package://det_perception_core/meshes/textured.dae";
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = pose.getOrigin().x();
            marker.pose.position.y = pose.getOrigin().y();
            marker.pose.position.z = pose.getOrigin().z();
            marker.pose.orientation.x = pose.getRotation().x();
            marker.pose.orientation.y = pose.getRotation().y();
            marker.pose.orientation.z = pose.getRotation().z();
            marker.pose.orientation.w = pose.getRotation().w();
            marker.scale.x = 1.0 * scale;
            marker.scale.y = 1.0 * scale;
            marker.scale.z = 1.0 * scale;
            marker.color.a = 1.0;
            marker.mesh_use_embedded_materials = true;
            m_marker_pub.publish(marker);
            clusters_curr.push_back(associated_cluster);
        }
        else {
            auto cluster_cloud = cluster_clouds[i];
            Eigen::Vector3f position;
            Eigen::Quaternionf orientation;
            Eigen::Vector3f dimensions;
            computeOBB<pcl::PointXYZ>(cluster_cloud, position, orientation, dimensions);
            // compute the homogeneous transformation matrix
            Eigen::Matrix4f Tworld_center = Eigen::Matrix4f::Identity();
            Tworld_center.block<3, 3>(0, 0) = orientation.toRotationMatrix();
            Tworld_center.block<3, 1>(0, 3) = position;
            auto Tcenter_origin = m_transform.inverse();
            auto Tworld_origin = Tworld_center * Tcenter_origin;
            // // publish a tf
            // tf::Transform transform;
            // transform.setOrigin(tf::Vector3(position[0], position[1], position[2]));
            // tf::Quaternion q(orientation.x(), orientation.y(), orientation.z(), orientation.w());
            // transform.setRotation(q);
            // m_br.sendTransform(tf::StampedTransform(transform, msg->header.stamp, msg->header.frame_id, 
            // "cluster_" + std::to_string(i)));
            // publish a marker
            visualization_msgs::Marker marker;
            marker.header.frame_id = msg->header.frame_id;
            marker.header.stamp = msg->header.stamp;
            marker.ns = "cluster_" + std::to_string(i);
            marker.id = 0;
            marker.type = visualization_msgs::Marker::MESH_RESOURCE;
            marker.mesh_resource = "package://det_perception_core/meshes/textured.dae";
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = Tworld_origin(0, 3);
            marker.pose.position.y = Tworld_origin(1, 3);
            marker.pose.position.z = Tworld_origin(2, 3);
            // convert the rotation matrix to quaternion
            Eigen::Quaternionf q2(Tworld_origin.block<3, 3>(0, 0));
            marker.pose.orientation.x = q2.x();
            marker.pose.orientation.y = q2.y();
            marker.pose.orientation.z = q2.z();
            marker.pose.orientation.w = q2.w();
            auto scale = std::max({dimensions[0], dimensions[1], dimensions[2]}) / 
            std::max({m_dimensions[0], m_dimensions[1], m_dimensions[2]});
            scale *= 0.90;
            marker.scale.x = 1.0 * scale;
            marker.scale.y = 1.0 * scale;    // init m_clusters_prev as an empty vector
            marker.scale.z = 1.0 * scale;
            marker.color.a = 1.0;
            marker.mesh_use_embedded_materials = true;
            m_marker_pub.publish(marker);
            // save the cluster
            Cluster cluster;
            cluster.pixel_center = centroid;
            tf::Transform pose;
            pose.setOrigin(tf::Vector3(Tworld_origin(0, 3), Tworld_origin(1, 3), Tworld_origin(2, 3)));
            tf::Quaternion q(q2.x(), q2.y(), q2.z(), q2.w());
            pose.setRotation(q);
            cluster.pose = pose;
            cluster.scale = scale;
            clusters_curr.push_back(cluster);
        }
    }
    m_clusters_prev = clusters_curr;
    // delete the markers that are not used anymore
    for (int i = cluster_clouds.size(); i < m_num_clusters_prev; i++)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = msg->header.frame_id;
        marker.header.stamp = msg->header.stamp;
        marker.ns = "cluster_" + std::to_string(i);
        marker.id = 0;
        marker.action = visualization_msgs::Marker::DELETE;
        m_marker_pub.publish(marker);
    }
    m_num_clusters_prev = cluster_clouds.size();
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
    m_raw_image = cv_ptr->image;

    // get foreground mask
    m_foreground_image_mask = imageBackgroundSubtraction(m_raw_image, m_background_image, 60);

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
    for (size_t i = 0; i < inliers->indices.size(); i++) {
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
    // if the image is grayscale, convert it to RGB
    if (image.channels() == 1) {
        cv::cvtColor(image, image_rgb, cv::COLOR_GRAY2RGB);
    }
    // if the image is already RGB, copy it
    else if (image.channels() == 3) {
        image_rgb = image.clone();
    }
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

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> PerceptionCore::getClusterClouds(
const typename OrderedCloud<pcl::PointXYZRGB>::Ptr ordered_cloud,
const cv::Mat& downsampled_mask, const int& num_inferences) {
    // get the point clouds of each cluster
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_clouds(num_inferences);
    for (int i = 0; i < num_inferences; i++) {
        cluster_clouds[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    }
    for (size_t i = 0; i < ordered_cloud->cloud->height; i++) {
    for (size_t j = 0; j < ordered_cloud->cloud->width; j++) {
        int x = ordered_cloud->start_x + j;
        int y = ordered_cloud->start_y + i;
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
        cluster_clouds[downsampled_mask.at<uchar>(y, x) - 1]->points.push_back(point);
    }
    }
    for (int i = 0; i < num_inferences; i++) {
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

std::vector<cv::Rect> PerceptionCore::expandBoundingBoxes(const std::vector<cv::Rect>& bboxes, const int& pixels) {
    // expand bounding boxes
    std::vector<cv::Rect> expanded_bboxes;
    for (size_t i = 0; i < bboxes.size(); i++) {
        cv::Rect bbox = bboxes[i];
        bbox.x -= pixels;
        bbox.y -= pixels;
        bbox.width += 2 * pixels;
        bbox.height += 2 * pixels;
        expanded_bboxes.push_back(bbox);
    }
    return expanded_bboxes;
}

cv::Mat PerceptionCore::mergeMasks(const cv::Mat& foreground_mask, const std::vector<unsigned char>& masks, 
const int& width, const int& height, const int& num_labels) {
    // merge masks
    cv::Mat merged_mask = cv::Mat::zeros(height, width, CV_8UC1);
    for (int n = 0; n < num_labels; n++) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
        if (masks[n * height * width + i * width + j] == 0) {
            continue;
        }
        if (foreground_mask.at<uchar>(i, j) == 0) {
            continue;
        }
        merged_mask.at<uchar>(i, j) = n + 1;
        }
    }
    }
    return merged_mask;
}

void PerceptionCore::loadMesh(const std::string& filename, pcl::PolygonMesh& mesh, Eigen::Matrix4f& transform,
Eigen::Vector3f& dimensions) {
    // load mesh
    pcl::io::loadPolygonFileSTL(filename, mesh);
    // compute the bounding box
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();
    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    Eigen::Vector3f position = position_OBB.getVector3fMap();
    Eigen::Quaternionf orientation = Eigen::Quaternionf(rotational_matrix_OBB);
    dimensions = (max_point_OBB.getVector3fMap() - min_point_OBB.getVector3fMap()).cwiseAbs();
    // compute the transform
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation(0, 3) = position[0];
    translation(1, 3) = position[1];
    translation(2, 3) = position[2];
    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    rotation.block<3, 3>(0, 0) = orientation.toRotationMatrix();
    transform = translation * rotation;
}

double PerceptionCore::computeClusterDiff(const cv::Mat &prev_mask, const cv::Mat &curr_mask, const cv::Rect &bbox)
{
    // compute the difference between the previous and current cluster masks
    int diff_pixels = 0;
    int total_pixels = 0;
    for (int i = bbox.y; i < bbox.y + bbox.height; i++) {
    for (int j = bbox.x; j < bbox.x + bbox.width; j++) {
        if (prev_mask.at<uchar>(i, j) != curr_mask.at<uchar>(i, j)) {
            diff_pixels++;
        }
        if (curr_mask.at<uchar>(i, j) != 0) {
            total_pixels++;
        }
    }
    }
    return (double)diff_pixels / (double)total_pixels;
}

bool PerceptionCore::clusterDiffStateMachine(const cv::Mat &prev_mask, const cv::Mat &curr_mask, 
const std::vector<cv::Rect> &bboxes, const double &threshold, std::vector<bool> &diffs) {
    bool need_inference = false;
    for (size_t i = 0; i < bboxes.size(); i++) {
        double diff = computeClusterDiff(prev_mask, curr_mask, bboxes[i]);
        if (diff > threshold) {
            need_inference = true;
            diffs[i] = true;
        } else {
            diffs[i] = false;
        }
    }
    return need_inference;
}

cv::Mat PerceptionCore::computeStationaryMovingAreas(const cv::Mat &prev_mask, const cv::Mat &curr_mask,
const std::vector<cv::Rect> &bboxes, const double &threshold) {
    // compute the stationary moving areas
    // empty == 0, stationary == 127, moving == 255
    cv::Mat stationary_moving_areas = cv::Mat::zeros(prev_mask.rows, prev_mask.cols, CV_8UC1);
    for (size_t i = 0; i < bboxes.size(); i++) {
        double diff = computeClusterDiff(prev_mask, curr_mask, bboxes[i]);
        if (diff > threshold) {
            stationary_moving_areas(bboxes[i]) = 255;
            // for (int j = bboxes[i].y; j < bboxes[i].y + bboxes[i].height; j++) {
            // for (int k = bboxes[i].x; k < bboxes[i].x + bboxes[i].width; k++) {
            //     if (prev_mask.at<uchar>(j, k) != 0) {
            //         stationary_moving_areas.at<uchar>(j, k) = 255;
            //     }
            // }
            // }
        }
        else {
            stationary_moving_areas(bboxes[i]) = 127;
            // for (int j = bboxes[i].y; j < bboxes[i].y + bboxes[i].height; j++) {
            // for (int k = bboxes[i].x; k < bboxes[i].x + bboxes[i].width; k++) {
            //     if (prev_mask.at<uchar>(j, k) != 0) {
            //         stationary_moving_areas.at<uchar>(j, k) = 127;
            //     }
            // }
            // }
        }
    }
    return stationary_moving_areas;
}

cv::Mat PerceptionCore::updateForegroundMask(const cv::Mat &foreground_mask, const std::vector<bool> &diffs,
const std::vector<cv::Rect> &bboxes) {
    // update the foreground mask
    cv::Mat updated_foreground_mask = foreground_mask.clone();
    for (size_t i = 0; i < bboxes.size(); i++) {
        if (diffs[i] == false) {
            for (int j = bboxes[i].y; j < bboxes[i].y + bboxes[i].height; j++) {
            for (int k = bboxes[i].x; k < bboxes[i].x + bboxes[i].width; k++) {
                updated_foreground_mask.at<uchar>(j, k) = 0;
            }
            }
        }
    }
    return updated_foreground_mask;
}

cv::Mat PerceptionCore::updateForegroundMask(const cv::Mat &foreground_mask, const std::vector<bool> &diffs,
const cv::Mat &labels) {
    // update the foreground mask
    cv::Mat updated_foreground_mask = foreground_mask.clone();
    for (int i = 0; i < labels.rows; i++) {
    for (int j = 0; j < labels.cols; j++) {
        if (labels.at<int>(i, j) == 0) {
            continue;
        }
        if (diffs[labels.at<int>(i, j) - 1] == false) {
            updated_foreground_mask.at<uchar>(i, j) = 0;
        }
    }
    }
    return updated_foreground_mask;
}

cv::Mat PerceptionCore::splitMask(const cv::Mat &foreground_mask, const std::vector<bool> &diffs, 
const cv::Mat &labels) {
    // split the foreground mask
    cv::Mat split_foreground_mask = foreground_mask.clone();
    for (int i = 0; i < labels.rows; i++) {
    for (int j = 0; j < labels.cols; j++) {
        if (labels.at<int>(i, j) == 0) {
            continue;
        }
        if (diffs[labels.at<int>(i, j) - 1] == false) {
            split_foreground_mask.at<uchar>(i, j) *= -1;
        }
    }
    }
    return split_foreground_mask;
}

std::vector<std::pair<int, int>> PerceptionCore::computeClusterPixelCentroids(const cv::Mat &merged_mask, 
const int &num_inferences) {
    // compute the centroids of the clusters
    std::vector<std::pair<int, int>> centroids(num_inferences);
    std::vector<int> num_pixels(num_inferences);
    for (int i = 0; i < merged_mask.rows; i++) {
    for (int j = 0; j < merged_mask.cols; j++) {
        if (merged_mask.at<uchar>(i, j) == 0) {
            continue;
        }
        centroids[merged_mask.at<uchar>(i, j) - 1].first += i;
        centroids[merged_mask.at<uchar>(i, j) - 1].second += j;
        num_pixels[merged_mask.at<uchar>(i, j) - 1]++;
    }
    }
    for (int i = 0; i < num_inferences; i++) {
        centroids[i].first /= num_pixels[i];
        centroids[i].second /= num_pixels[i];
    }
    return centroids;
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
