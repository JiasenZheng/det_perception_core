/**
 * @file colorization.cpp
 * @author Jiasen Zheng
 * @brief colorize the raw point cloud with the color image
 */

#include "colorization.hpp"

Colorization::Colorization(ros::NodeHandle nh): 
m_nh(nh)
{
    m_nh.getParam("/colorization/cam", m_cam_name);
    m_nh.getParam("/colorization/lidar", m_lidar_name);
    m_sub_pointcloud = m_nh.subscribe("/" + m_lidar_name + "/livox/lidar", 1, &Colorization::pointcloudCallback, this);
    m_sub_image = m_nh.subscribe("/" + m_cam_name + "/camera/color/image_raw", 1, &Colorization::imageCallback, this);
    m_pub_pointcloud = m_nh.advertise<sensor_msgs::PointCloud2>("/" + m_lidar_name + "/livox/colorized", 1);
    m_intrinsic = cv::Mat::zeros(3, 3, CV_64F);
    m_extrinsic = cv::Mat::zeros(4, 4, CV_64F);
    // get extrinsic from parameter server
    std::vector<double> extrinsic;
    m_nh.getParam("/" + m_cam_name + "/extrinsic", extrinsic);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m_extrinsic.at<double>(i, j) = extrinsic[i * 4 + j];
        }
    }
    // get intrinsic from parameter server
    std::vector<double> intrinsic;
    m_nh.getParam("/" + m_cam_name + "/intrinsic", intrinsic);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            m_intrinsic.at<double>(i, j) = intrinsic[i * 3 + j];
        }
    }
}

void Colorization::run()
{

}

void Colorization::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colorizeCloud(m_image, cloud, colorized_cloud, m_intrinsic, m_extrinsic);
    sensor_msgs::PointCloud2 colorized_msg;
    pcl::toROSMsg(*colorized_cloud, colorized_msg);
    colorized_msg.header.frame_id = msg->header.frame_id;
    colorized_msg.header.stamp = msg->header.stamp;
    m_pub_pointcloud.publish(colorized_msg);
}

void Colorization::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
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
    m_image = cv_ptr->image;
}

void Colorization::parseMatrix(const std::string& path, cv::Mat& matrix)
{
    std::ifstream file(path);
    std::string line;
    int row = 0;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string token;
        int col = 0;
        while (std::getline(iss, token, ','))
        {
            matrix.at<double>(row, col) = std::stod(token);
            col++;
        }
        row++;
    }
}

void Colorization::colorizeCloud(const cv::Mat& image, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colorized_cloud, const cv::Mat& intrinsic, const cv::Mat& extrinsic)
{
    std::vector<cv::Point3f> pts_3d;
    for (size_t i = 0; i < cloud->points.size(); i++)
    {
        pcl::PointXYZI pt = cloud->points[i];
        pts_3d.push_back(cv::Point3f(pt.x, pt.y, pt.z));
    }
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
    double t_x, t_y, t_z;
    t_x = extrinsic.at<double>(0, 3);
    t_y = extrinsic.at<double>(1, 3);
    t_z = extrinsic.at<double>(2, 3);
    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << t_x, t_y, t_z);
    cv::Mat rot_mat = extrinsic(cv::Rect(0, 0, 3, 3));
    cv::Mat r_vec;
    cv::Rodrigues(rot_mat, r_vec);
    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(pts_3d, r_vec, t_vec, intrinsic, dist, pts_2d);
    int width = image.cols;
    int height = image.rows;
    for (size_t i = 0; i < pts_2d.size(); i++)
    {
        cv::Point2f pt = pts_2d[i];
        if (pt.x < 0 || pt.x >= width || pt.y < 0 || pt.y >= height)
        {
            continue;
        }
        pcl::PointXYZI pt_i = cloud->points[i];
        pcl::PointXYZRGB pt_rgb;
        pt_rgb.x = pt_i.x;
        pt_rgb.y = pt_i.y;
        pt_rgb.z = pt_i.z;
        pt_rgb.r = image.at<cv::Vec3b>(pt.y, pt.x)[2];
        pt_rgb.g = image.at<cv::Vec3b>(pt.y, pt.x)[1];
        pt_rgb.b = image.at<cv::Vec3b>(pt.y, pt.x)[0];
        colorized_cloud->points.push_back(pt_rgb);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "colorization");
    ros::NodeHandle nh;
    Colorization colorization(nh);
    colorization.run();
    ros::spin();
    return 0;
}
