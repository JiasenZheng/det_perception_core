/**
 * @file colorization.cpp
 * @author Jiasen Zheng
 * @brief colorize the raw point cloud with the color image
 */

#include "colorization.hpp"

Colorization::Colorization(ros::NodeHandle nh)
{
    m_nh = nh;
    // subscribe to the point cloud and the color image
    m_sub_pointcloud = m_nh.subscribe("/livox/lidar", 1, &Colorization::pointcloudCallback, this);
    m_sub_image = m_nh.subscribe("/camera/color/image_raw", 1, &Colorization::imageCallback, this);
    // publish colorized point cloud
    m_pub_pointcloud = m_nh.advertise<sensor_msgs::PointCloud2>("/livox/colorized", 1);
}

Colorization::~Colorization()
{

}

void Colorization::run()
{

}

void Colorization::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud);
    // convert to message
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = msg->header.frame_id;
    cloud_msg.header.stamp = msg->header.stamp;
    // publish
    m_pub_pointcloud.publish(cloud_msg);
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
