#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include "gs_common.h"
#include <scan_img/RobotState.h>
#include <nav_msgs/Odometry.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::LaserScan, nav_msgs::Odometry> MySyncPolicy;

class LocalMap
{
public:
    LocalMap(){}
    void init(int size_x, int size_y, float delta)
    {
        delta_ = delta;
        data_ = new unsigned char[size_x * size_y];
        size_x_ = size_x;
        size_y_ = size_y;
        tf_map_base_.setOrigin(tf::Vector3(size_x_ * delta_ / 2.0, size_y_ * delta_ / 2.0, 0));
        tf::Quaternion q;
        q.setEuler(3.14159, 0, -1.5708);
        tf_map_base_.setRotation(q);
        tf_base_map_ = tf_map_base_.inverse();
        clear();
    }
    void clear()
    {
        //unknown
        for (int i = 0; i < size_x_ * size_y_; i++)
            data_[i] = 100;
    }
    bool free()
    {
        delete data_;
        data_ = NULL;
    }
    ~LocalMap()
    {
        free();
    }
    bool getIndex(float p_x, float p_y, int& x, int& y)
    {
        tf::Vector3 p_base(p_x, p_y, 0);
        tf::Vector3 p_map = tf_base_map_ * p_base;
        x = p_map.getX() / delta_;
        y = p_map.getY() / delta_;
        if (check_in(x, y))
        {
            return true;
        }
        else
            return false;
    }
    bool check_in(int x, int y)
    {
        return (x >= 0 && x < size_y_ && y >= 0 && y < size_x_);
    }
    int getDataId(int x, int y)
    {
        return (y*size_y_ + x);
    }
    void getDataIndex(int d, int& x, int& y)
    {
        x = d % size_y_;
        y = d / size_y_;
    }
    void mapToBase(int x, int y, float& p_x, float& p_y)
    {
        tf::Vector3 p_map(x * delta_, y * delta_, 0);
        tf::Vector3 p_base = tf_map_base_ * p_map;
        p_x = p_base.getX();
        p_y = p_base.getY();
    }
    unsigned char* data_;
    float delta_;
    int size_x_;
    int size_y_;
    tf::Transform tf_base_map_;
    tf::Transform tf_map_base_;
};

class ScanImage
{
public:
    ScanImage();
    ~ScanImage();
private:
    ros::NodeHandle nh_;
    ros::Subscriber goal_sub_;

    void goalCallback(const geometry_msgs::PoseStamped &msg);
    std::string min_dist_topic_;

    bool dealScan(const sensor_msgs::LaserScan &scan, scan_img::RobotState &state_msg, tf::Transform& tf_goal_robot);

    ros::Publisher image_pub_;
    ros::Publisher state_pub_;
    ros::Publisher min_dist_publisher_;
    LocalMap local_map_;
    std::string laser_frame_;

    //for testing
    ros::Subscriber odom_sub;
    void callback_odom(const nav_msgs::Odometry &msg);
    ros::Subscriber scan_sub;
    void callback_scan(const sensor_msgs::LaserScan &msg);

    message_filters::Subscriber<sensor_msgs::LaserScan> *scanSubscriber_;
    message_filters::Subscriber<nav_msgs::Odometry> *odomSubscriber_;
    message_filters::Synchronizer<MySyncPolicy> *syn_;
    void laser_odom_callback(const sensor_msgs::LaserScanConstPtr &laser, const nav_msgs::OdometryConstPtr &odom);
    float getFooprintDist(float th);

    int image_width_;
    int image_height_;
    double resolution_;
    double max_dis_;
    std::string laser_topic_;
    std::string goal_topic_;
    std::string image_topic_;
    std::string odom_topic_;
    std::string state_topic_;
    std::string map_topic_;
    std::string base_frame_;
    std::string footprint_frame_;
    std::string robot_name_;
    double angle_min_;
    double angle_max_;
    double rob_len_;
    double rob_wid_;

    tf::Stamped<tf::Pose> tf_laser_base_;
    tf::Stamped<tf::Pose> tf_goal_world_;
    tf::Stamped<tf::Pose> tf_robot_world_;
    double goal_tolerance_;
    double collision_th_;
};
