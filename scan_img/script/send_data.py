#!/usr/bin/env python3
import rospy
#import tf
import tf2_ros
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3


if __name__ == '__main__':
    rospy.init_node('tf_listener')
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    #listener = tf.TransformListener()
    pub = rospy.Publisher('lscm/odom', Odometry, queue_size=1)
    rate = rospy.Rate(10.0)
        
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('world', 'agv_base_link', rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue
        
        current_time = rospy.Time.now()
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.pose.pose = Pose(Point(trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z), Quaternion(trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w))

        odom.child_frame_id = "base_link"
        #odom.twist.twist = Twist(Vector3(vx, vy, 0), Vector3(0, 0, vth))
        #angular = 4 * math.atan2(trans[1], trans[0])
        #linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2)
        #cmd = geometry_msgs.msg.Twist()
        #cmd.linear.x = linear
        #cmd.angular.z = angular
        odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        pub.publish(odom)
        rate.sleep()
