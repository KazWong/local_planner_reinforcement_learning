import ros_utils as util

import yaml
import os
import cv2

import rospy
from geometry_msgs.msg import Twist, PoseStamped
from scan_img.msg import RobotState
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class Env(object):
    def __init__(self, cfg_names):
        self.cfg_names = cfg_names
        self.control_hz = 0.2
        self.collision_th = 0.5
        self.image_size = (60, 60)
        self.image_batch = 1
        self.done = 0
        self.epoch = 0
        self.reset_count = 0
        self.state_last = None
        self.discrete_actions = []

        self.robot_model_name = ""
        self.robot_radius = 0.0

        rospy.init_node("simulator_ros")

        self.bridge = CvBridge()

        self.read_yaml(cfg_names[0])
        self.vel_pub = rospy.Publisher( '/' + self.robot_name + '/cmd_vel', Twist, queue_size=1)
        self.goal_pub = rospy.Publisher( '/' + self.robot_name + '/goal', PoseStamped, queue_size=1)
        self.state_sub = rospy.Subscriber("/" + self.robot_name + "/state", RobotState, self.state_callback, queue_size=1)

    def set_img_size(self, img_size):
        self.image_size = img_size
        self.init()

    def set_colis_dist(self, dist):
        self.collision_th = dist

    def read_yaml(self, yaml_file):
        pkg_path = util.get_pkg_path('gz_pkg')
        final_file = os.path.join(pkg_path, 'drl', 'cfg', yaml_file)
        #final_file = os.path.join(pkg_path, '..', 'drl', 'cfg', yaml_file)
        with open(final_file, 'r') as f:
            self.envs_cfg = yaml.load(f)
        self.robot_radius = self.envs_cfg['robot_radius']
        self.robot_name = self.envs_cfg['robot_model_name']

    def image_trans(self, img_ros):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(img_ros, desired_encoding="passthrough")
        except CvBridgeError as e:
          print(e)
        image = cv2.resize(cv_image, self.image_size) / 255.0
        return image

    def robot_control_original(self, action):
        vel = Twist()

        if self.done == 0:
            vel.linear.x = action[0]
            vel.angular.z = action[1]
        else:
            vel.linear.x = 0
            vel.angular.z = 0
        self.vel_pub.publish(vel)

    def publish_goal(self, target_poses):
        #send goal in the odom frame
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.robot_name + "/odom"
        goal_msg.header.stamp = rospy.Time.now()

        goal_msg.pose.position.x = self.target_poses[0][0]
        goal_msg.pose.position.y = self.target_poses[0][1]
        rpy = [0, 0, self.target_poses[0][2]]
        q = util.rpy_to_q(rpy)
        goal_msg.pose.orientation.x = q[0]
        goal_msg.pose.orientation.y = q[1]
        goal_msg.pose.orientation.z = q[2]
        goal_msg.pose.orientation.w = q[3]
        self.goal_pub.publish(goal_msg)

    def state_callback(self, msg):
        if self.robot_name in msg.laser_image.header.frame_id:
            self.state_last = msg # sub msg from ros

    def get_states(self):
        raise NotImplementedError()

    def get_rewards(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step_discrete(self, action):
        raise NotImplementedError()
