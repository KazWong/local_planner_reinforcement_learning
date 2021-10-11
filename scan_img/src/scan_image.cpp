#include "scan_image.h"
#include <cmath>

ScanImage::ScanImage() :
    scanSubscriber_(NULL), odomSubscriber_(NULL), syn_(NULL)
{
    ros::NodeHandle private_nh("~");
    private_nh.param("robot_name", robot_name_, std::string("lscm"));
    private_nh.param("image_width", image_width_, 60); 
    private_nh.param("image_height", image_height_, 60);
    private_nh.param("resolution", resolution_, 0.1);
    private_nh.param("laser_topic", laser_topic_, std::string("scan"));
    private_nh.param("odom_topic", odom_topic_, std::string("odom"));
    private_nh.param("state_topic", state_topic_, std::string("state"));
    private_nh.param("goal_topic", goal_topic_, std::string("goal"));
    private_nh.param("image_topic", image_topic_, std::string("scan_image"));
    private_nh.param("min_dist_topic", min_dist_topic_, std::string("min_dist"));

    private_nh.param("base_frame", base_frame_, std::string("base_link"));
    private_nh.param("footprint_frame", footprint_frame_, std::string("base_footprint"));
    private_nh.param("laser_frame", laser_frame_, std::string("laser_link"));
    private_nh.param("angle_min", angle_min_, -4.0);
    private_nh.param("angle_max", angle_max_, 4.0);
    private_nh.param("goal_tolerance", goal_tolerance_, 0.3);
    private_nh.param("collision_th", collision_th_, 0.5);
    private_nh.param("rob_len", rob_len_, 0.6);
    private_nh.param("rob_wid", rob_wid_, 0.4);
    local_map_.init(image_width_,image_height_,resolution_);
    double width = image_width_ * resolution_;
    double height = image_height_ * resolution_;
    max_dis_ = sqrt(width * width / 4.0 + height * height / 4.0);

    odomSubscriber_ = new message_filters::Subscriber<nav_msgs::Odometry>(nh_, odom_topic_, 1);
    scanSubscriber_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh_, laser_topic_, 1);
    syn_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *scanSubscriber_, *odomSubscriber_);
    syn_->registerCallback(boost::bind(&ScanImage::laser_odom_callback, this, _1, _2));

    //odom_sub = nh_.subscribe("/turtlerobot0/odom", 1, &ScanImage::callback_odom, this);
    //scan_sub = nh_.subscribe("/turtlerobot0/scan", 1, &ScanImage::callback_scan, this);

    goal_sub_ = nh_.subscribe(goal_topic_, 1, &ScanImage::goalCallback, this);
    image_pub_ = nh_.advertise<sensor_msgs::Image>(image_topic_, 1);
    state_pub_ = nh_.advertise<scan_img::RobotState>(state_topic_, 1);
    min_dist_publisher_ = nh_.advertise<geometry_msgs::PointStamped>(min_dist_topic_, 1);
    //set the tf for lidar and base
    tf_laser_base_.setOrigin(tf::Vector3(0.0, 0.0, 0.5));
    tf_laser_base_.setRotation(tf::Quaternion(0.0, 0.0, 0.0, 1.0));
}

bool ScanImage::dealScan(const sensor_msgs::LaserScan &scan, scan_img::RobotState &state_msg, tf::Transform& tf_goal_robot)
{
    //ROS_INFO("in ScanImage::dealScan-------------------------------");
    local_map_.clear();
    laser_frame_ = scan.header.frame_id;
    state_msg.min_dist.header = scan.header;
    state_msg.min_dist.header.frame_id = base_frame_;
    state_msg.min_dist.point.z = 999;
    IntPoint p0;
    tf::Vector3 xy_p0, xy_p0_base;
    xy_p0.setValue(0, 0, 0);
    xy_p0_base = tf_laser_base_ * xy_p0;
    local_map_.getIndex(xy_p0_base.getX(), xy_p0_base.getY(), p0.x, p0.y);
    
    state_msg.collision = 0;
    for (int i = 0; i < scan.ranges.size(); i++)
    {
        double d;
        if (std::isnan(scan.ranges[i]))
            continue;
        else if (std::isinf(scan.ranges[i]))
            d = max_dis_;
        else
            d = scan.ranges[i];
        double theta = scan.angle_min + scan.angle_increment * i;
        if (theta < angle_min_ || theta > angle_max_)
            continue;
        double x = d * cos(theta);
        double y = d * sin(theta);
        tf::Vector3 xy, xy_base;
        xy.setValue(x, y, 0);
        xy_base = tf_laser_base_ * xy;
        double x_base = xy_base.getX();
        double y_base = xy_base.getY();
        
        //penalty for min distance
        double dist = sqrt(x_base * x_base + y_base * y_base);
        if (dist < state_msg.min_dist.point.z)
        {
            state_msg.min_dist.point.x = x_base;
            state_msg.min_dist.point.y = y_base;
            state_msg.min_dist.point.z = dist;
        }

        //penalty for collision
        float x_dist = x_base - getFooprintDist(theta) * cos(theta);
        float y_dist = y_base - getFooprintDist(theta) * sin(theta);
        float c_dist = sqrt(x_dist*x_dist + y_dist*y_dist);
        //ROS_INFO("collision: %f, %f, %f, %f, %f, %f", theta, x_base, y_base, x_dist, y_dist, c_dist);
        if(c_dist < 0.1)
            state_msg.collision = 1;

        IntPoint p1;
        local_map_.getIndex(x_base,y_base, p1.x, p1.y);
        IntPoint linePoints[2000];
        GridLineTraversalLine line;
        line.points = linePoints;
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p1, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                //free
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] = 255;
            }
        }
        if (local_map_.check_in(p1.x, p1.y))
        {
            //obstacle
            local_map_.data_[local_map_.getDataId(p1.x, p1.y)] = 0;
        }
    }

    min_dist_publisher_.publish(state_msg.min_dist);
    state_msg.laser_image.encoding = "8UC1";
    state_msg.laser_image.height = image_height_;
    state_msg.laser_image.width = image_width_;
    state_msg.laser_image.step = image_height_ * 1;
    state_msg.laser_image.header = scan.header;
    state_msg.laser_image.header.frame_id = base_frame_;
    int image_size = image_height_ * image_width_;
    state_msg.laser_image.data.resize(image_size);
    
    for (int m = 0; m < image_size; m++)
    {
        state_msg.laser_image.data[m] = local_map_.data_[m];
    }
    image_pub_.publish(state_msg.laser_image);

    state_msg.laser = scan;

    return true;
}

//void ScanImage::callback_odom(const nav_msgs::Odometry &msg)
//{
//    ROS_INFO("in ScanImage::callback_odom: %f", msg.header.stamp.toSec());
//}

//void ScanImage::callback_scan(const sensor_msgs::LaserScan &msg)
//{
//    ROS_INFO("in ScanImage::callback_scan: %f", msg.header.stamp.toSec());
//}

void ScanImage::laser_odom_callback(const sensor_msgs::LaserScanConstPtr &laser, const nav_msgs::OdometryConstPtr &odom)
{
    //ROS_INFO("in ScanImage::laser_odom_callback");
    scan_img::RobotState state_msg;

    //robot to goal in robot frame
    tf_robot_world_.setOrigin(tf::Vector3(odom->pose.pose.position.x, odom->pose.pose.position.y, 0));
    tf_robot_world_.setRotation(tf::Quaternion(odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z, odom->pose.pose.orientation.w));
    tf::Transform tf_goal_robot = tf_robot_world_.inverse() * tf_goal_world_;
    state_msg.pose.position.x = tf_goal_robot.getOrigin().getX();
    state_msg.pose.position.y = tf_goal_robot.getOrigin().getY();
    state_msg.pose.position.z = tf_goal_robot.getOrigin().getZ();
    state_msg.pose.orientation.x = tf_goal_robot.getRotation().getX();
    state_msg.pose.orientation.y = tf_goal_robot.getRotation().getY();
    state_msg.pose.orientation.z = tf_goal_robot.getRotation().getZ();
    state_msg.pose.orientation.w = tf_goal_robot.getRotation().getW();

    state_msg.vel = odom->twist.twist;
    
    bool success = dealScan(*laser, state_msg, tf_goal_robot);
    if (success)
    {
        state_pub_.publish(state_msg);
    }
}

void ScanImage::goalCallback(const geometry_msgs::PoseStamped &msg)
{
    //ROS_INFO("in ScanImage::goalCallback");
    tf_goal_world_.setOrigin(tf::Vector3(msg.pose.position.x, msg.pose.position.y, 0));
    tf_goal_world_.setRotation(tf::Quaternion(0, 0, 0, 1));
}

float ScanImage::getFooprintDist(float th)
{
    float mag = 0;
    float abs_cos_angle= fabs(cos(th));
    float abs_sin_angle= fabs(sin(th));
    if(rob_len_/2*abs_sin_angle <= rob_wid_/2*abs_cos_angle)
    {
        mag = rob_len_/2/abs_cos_angle;
    }
    else
    {
        mag = rob_wid_/2/abs_sin_angle;
    }

    return mag;
}

ScanImage::~ScanImage()
{
    if (syn_ != NULL)
    {
        delete syn_;
        syn_ = NULL;
    }
    if (scanSubscriber_ != NULL)
    {
        delete scanSubscriber_;
        scanSubscriber_ = NULL;
    }
    if (odomSubscriber_ != NULL)
    {
        delete odomSubscriber_;
        odomSubscriber_ = NULL;
    }
}
