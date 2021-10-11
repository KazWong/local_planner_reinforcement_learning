from env import Env
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped, Point, PoseStamped
from std_msgs.msg import Int8
import math
import ros_utils as util
import random
import numpy as np
import time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import copy
from gz_ros import *
import yaml
import os
from scan_img.msg import RobotState

class GazeboEnv(Env):
    def __init__(self, cfg_names):
        rospy.init_node("drl")
        self.epoch = 0
        self.cfg_names = cfg_names
        self.read_yaml(cfg_names[0])
        self.collision_th = 0.5
        self.control_hz = 0.2
        self.image_size = (60, 60)
        self.image_batch = 1
        self.init_datas()

        self.reset_count = 0
        #no use
        self.step_count = 0

        self.discrete_actions = \
        [[0.0, -0.9], [0.0, -0.6], [0.0, -0.3], [0.0, 0.05], [0.0, 0.3], [0.0, 0.6], [0.0, 0.9],
        [0.2, -0.9], [0.2, -0.6], [0.2, -0.3], [0.2, 0], [0.2, 0.3], [0.2, 0.6], [0.2, 0.9],
        [0.4, -0.9], [0.4, -0.6], [0.4, -0.3], [0.4, 0], [0.4, 0.3], [0.4, 0.6], [0.4, 0.9],
        [0.6, -0.9], [0.6, -0.6], [0.6, -0.3], [0.6, 0], [0.6, 0.3], [0.6, 0.6], [0.6, 0.9]]


        self.bridge = CvBridge()

        self.state_sub = rospy.Subscriber("/" + self.robot_name + "/state", RobotState, self.state_callback, queue_size=1)

    def get_states(self, save_img=None):
        states, images_last, min_dists, collisions, scans, vels = self.get_robots_state()

        #for debug
        if save_img != None:
            cv2.imwrite(save_img + "_robot" + ".png", images_last * 255)
        ptr = self.images_ptr[0]
        if ptr < self.image_batch:
            for j in range(ptr, self.image_batch):
                self.images_batch[j] = copy.deepcopy(images_last)
        else:
            self.images_batch[ptr % self.image_batch] = copy.deepcopy(images_last)
        self.images_ptr[0] += 1

        images_reshape = np.transpose(self.images_batch, (1, 2, 0))
        return (np.array(states), images_reshape, min_dists, collisions, scans, vels)

    #TODO: reward calculation, to be tuned
    def get_rewards(self, state, min_dist, is_collision):
        distance_reward_factor = 200
        obs_reward_factor = 100

        print("dist to obs: ", min_dist)
        reward = collision_reward = reach_reward = step_reward = distance_reward = 0
        done = 0
        if min_dist < 1.0:
            if self.last_d_obs == -1:
                self.last_d_obs = min_dist
                collision_reward = 0
            else:
                collision_reward = (min_dist - self.last_d_obs) * obs_reward_factor
                self.last_d_obs = min_dist

        print("collision: ", is_collision)

        #if min_dist <= self.collision_th:
        if is_collision:
            print("collision!!!!!!!!")
            done = -1
            collision_reward = -500

        d = math.sqrt(state[0] * state[0] + state[1] * state[1])
        print("dist to goal: ", d)
        if d < 0.3:
            print("arrive")
            reach_reward = 500
            done = -2
        else:
            if self.last_d == -1:
                self.last_d = d
                distance_reward = 0
            else:
                distance_reward = (self.last_d - d) * distance_reward_factor
                self.last_d = d
            step_reward = -5
            # distance_reward = (distance_reward_factor * 1/d)

        reward = collision_reward + reach_reward + step_reward + distance_reward
        print("reawrd is: ", reward)
        if done < 0 and self.done == 0:
            self.done = done
        return (reward, done)

    def reset(self, target_dist = 0.0):
        self.robot_control([0, 0])
        self.get_avoid_areas()
        time.sleep(0.5)
        self.reset_obs()
        time.sleep(0.5)
        self.reset_robots(target_dist=target_dist)
        time.sleep(1.0)

        self.last_d_obs = -1
        self.last_d = -1
        self.done = 0

        self.reset_count += 1
        self.step_count = 0
        return self.get_states()

    def step(self, action):
        self.robot_control(action)
        rospy.sleep(self.control_hz)
        self.step_count += 1
        states = self.get_states()
        rw = self.get_rewards(states[0], states[2], states[3])
        if rw == False:
            return False, False, False
        else:
            return states, np.array(rw[0], dtype='float64'), np.array(rw[1])

    def step_discrete(self, action):
        return self.step(self.discrete_actions[action])

    def image_trans(self, img_ros):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(img_ros, desired_encoding="passthrough")
        except CvBridgeError as e:
          print(e)
        image = cv2.resize(cv_image, self.image_size) / 255.0
        return image

    def state_callback(self, msg):
        if self.robot_name in msg.laser_image.header.frame_id:
            self.state_last = msg

    #TODO
    def get_avoid_areas(self):
        #obstacle cannot be placed on the robot
        self.obs_avoid_areas = []
        #if self.envs_cfg['begin_poses_type'][0] == 'fix':
        #    self.obs_avoid_areas.append(self.envs_cfg['begin_poses'][0][0:2] + [self.robot_radius])
        #if self.envs_cfg['target_poses_type'][0] == 'fix':
        #    self.obs_avoid_areas.append(self.envs_cfg['target_poses'][0][0:2] + [self.robot_radius])
        #robot cannot be placed on the obstacle
        self.rob_avoid_areas = []
        for i in range(len(self.envs_cfg['models'])):
            pose_range = self.envs_cfg['models_pose'][i]
            model_radius = self.envs_cfg['models_radius'][i]
            pose_type = self.envs_cfg['model_poses_type'][i]
            if pose_type == 'fix':
                self.rob_avoid_areas.append(pose_range[0:2] + [model_radius])

    def reset_robots(self, target_dist=0.0):
        self.init_poses = []
        self.target_poses = []

        #send goal in the odom frame
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.robot_name + "/odom"
        goal_msg.header.stamp = rospy.Time.now()

        #start
        while True:
            pose_range = self.envs_cfg['begin_poses'][0]
            rand_pose = self.random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
            if self.free_check_robot(rand_pose[0], rand_pose[1], self.init_poses) and self.free_check_obj([rand_pose[0], rand_pose[1], self.robot_radius], self.obs_range):
                self.init_poses.append(rand_pose[:])
                break

        if self.robot_name in get_world_models():
            set_model_state(self.robot_name, self.init_poses[0])
        else:
            print(self.robot_model_name)
            spawn_model(self.robot_name, get_model_sdf(self.robot_model_name), self.init_poses[0], self.robot_name)

        #goal
        while True:
            pose_range = self.envs_cfg['target_poses'][0]
            rand_pose = self.random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
            if (self.init_poses[0][0] - rand_pose[0]) ** 2 + (self.init_poses[0][1] - rand_pose[1]) ** 2 > target_dist ** 2 and self.free_check_robot(rand_pose[0], rand_pose[1], self.target_poses) and self.free_check_obj([rand_pose[0], rand_pose[1], self.robot_radius], self.obs_range):
                self.target_poses.append(rand_pose[:])
                break

        goal_msg.pose.position.x = self.target_poses[0][0]
        goal_msg.pose.position.y = self.target_poses[0][1]
        rpy = [0, 0, self.target_poses[0][2]]
        q = util.rpy_to_q(rpy)
        goal_msg.pose.orientation.x = q[0]
        goal_msg.pose.orientation.y = q[1]
        goal_msg.pose.orientation.z = q[2]
        goal_msg.pose.orientation.w = q[3]
        self.goal_pub.publish(goal_msg)

    def del_all_obs(self):
        for model_name in self.obs_name:
            if model_name!='room':
                delete_model(model_name)
        self.obs_name = []

    def reset_obs(self):
        self.obs_range = []
        for i in range(len(self.envs_cfg['models'])):
            pose_range = self.envs_cfg['models_pose'][i]
            model_radius = self.envs_cfg['models_radius'][i]
            pose_type = self.envs_cfg['model_poses_type'][i]
            model_name = self.envs_cfg['models_name'][i]

            if pose_type == 'fix':
                self.obs_range.append(pose_range + [0, model_radius])
            elif pose_type == 'range':
                while True:
                    rand_pose = self.random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
                    if self.free_check_obj([rand_pose[0], rand_pose[1], model_radius], self.obs_range) and self.free_check_obj([rand_pose[0], rand_pose[1], model_radius], self.obs_avoid_areas):
                        self.obs_range.append(rand_pose + [model_radius])
                        break
            if model_name in get_world_models():
                set_model_state(model_name, self.obs_range[i][0:3])
            else:
                self.obs_name.append(model_name)
                spawn_model(model_name, get_model_sdf(self.envs_cfg['models'][i]), self.obs_range[i][0:3])

    def get_robots_state(self):
        state = self.state_last
        image = self.image_trans(state.laser_image)
        goal_pose = [state.pose.position.x, state.pose.position.y]
        min_dist = state.min_dist.point.z
        is_collision = state.collision
        scan = state.laser
        vel = state.vel
        return goal_pose, image, min_dist, is_collision, scan, vel

    def random_pose(self, x, y, sita):
        return [random.uniform(x[0],x[1]), random.uniform(y[0],y[1]), random.uniform(sita[0],sita[1])]

    def free_check_robot(self, x, y, robot_poses):
        d = self.robot_radius*2
        for pose in robot_poses:
            test_d = math.sqrt((x-pose[0])*(x-pose[0]) + (y-pose[1])*(y-pose[1]))
            if test_d <= d:
                return False
        return True

    def free_check_obj(self, target_pose, obj_poses):
        for pose in obj_poses:
            if pose[-1] == 0.0:
                continue
            d = target_pose[-1] + pose[-1]
            test_d = math.sqrt((target_pose[0]-pose[0])**2 + (target_pose[1]-pose[1])**2)
            if test_d <= d:
                return False
        return True

    #remove?
    def random_robots_pose(self, pose_ranges):
        robot_poses = []
        pose_range = random.choice(pose_ranges)
        while True:
            if len(pose_range) == 4:
                rand_pose = self.random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
            elif len(pose_range) == 6:
                rand_pose = self.random_pose(pose_range[:2], pose_range[2:4], pose_range[4:6])
            if self.free_check_robot(rand_pose[0], rand_pose[1], robot_poses):
                robot_poses.append(rand_pose[:])
                break
        return robot_poses[:]

    def robot_control(self, action):
        vel = Twist()

        if self.done == 0:
            vel.linear.x = action[0]
            vel.angular.z = action[1]
        else:
            vel.linear.x = 0
            vel.angular.z = 0
        self.vel_pub.publish(vel)

    def get_robot_name(self, i):
        return self.robot_model_name

    #TODO:
    def empty_robots(self):
        model_names = get_world_models()
        x = 0
        y = 20
        if self.robot_name in model_names:
            x += 3 * self.robot_radius
            set_model_state(self.robot_name, [x, y])

    def set_img_size(self, img_size):
        self.image_size = img_size
        self.init()

    def set_colis_dist(self, dist):
        self.collision_th = dist

    def read_yaml(self, yaml_file):
        pkg_path = get_pkg_path('gz_pkg')
        #final_file = os.path.join(pkg_path, 'drl', 'cfg', yaml_file)
        final_file = os.path.join(pkg_path, '..', 'drl', 'cfg', yaml_file)
        with open(final_file, 'r') as f:
            self.envs_cfg = yaml.load(f)
        self.robot_radius = self.envs_cfg['robot_radius']
        self.robot_model_name = self.envs_cfg['robot_model_name']

    def init_datas(self):
        self.last_d = -1
        self.last_d_obs = -1
        self.images_batch = [None] * self.image_batch
        self.images_ptr = []
        self.state_last = None
        self.robot_name = self.get_robot_name(0)
        self.obs_name = []
        self.done = 0
        self.obstacles_ranges = []

        self.images_ptr.append(0)
        self.vel_pub = rospy.Publisher( '/' + self.robot_name + '/cmd_vel', Twist, queue_size=1)
        self.goal_pub = rospy.Publisher( '/' + self.robot_name + '/goal', PoseStamped, queue_size=1)
        self.obstacles_ranges.append([])

if __name__ == '__main__':
    env = GazeboEnv()
    rospy.sleep(0.5)
    # env.reset()
