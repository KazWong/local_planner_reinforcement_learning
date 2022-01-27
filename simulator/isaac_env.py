from env import Env
from isaac_ros_tf import *
import numpy as np
import random
import math
import os
from test_env import Env_config
from test_robot import Robot_config
from geometry_msgs.msg import Twist
#from isaac_ros_tf import Send_data
#from isaac_ros_tf2 import Send_data
from omni.isaac.python_app import OmniKitHelper
import copy
import time
import cv2
import threading
from threading import Thread

class IsaacEnv(Env):
	def __init__(self, cfg_names):
	    super().__init__(cfg_names)
	    self.startup_config = {"renderer": "RayTracedLighting", "headless": False, "experience": '/home/slam/.local/share/ov/pkg/isaac_sim-2021.1.1/apps/omni.isaac.sim.python.kit'}
	    self.kit = OmniKitHelper(self.startup_config)
	    import omni
	    from pxr import UsdGeom
	    self.omni = omni
	    self.image_batch = 1
	    self.init_datas()
	    self.test_env = Env_config(omni,self.kit)
	    self.stage = self.kit.get_stage()
	    self.test_rob = Robot_config(self.stage, self.omni)
	    self.send_data = Send_data()
	    import omni.isaac.dr as dr
	    self.dr_interface = dr._dr.acquire_dr_interface()
	    self.prefix = "/World/" + self.robot_name
	    #self.pose_differ = []
	    self.step_count = 0
	    self.convert_m = 100
	    self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.omni.usd.get_context().get_stage())
	    while self.kit.is_loading():
	        self.kit.update(1 / 60.0)
	    self.lv = 0
	    self.av = 0
	    self.discrete_actions = \
        [[0.0, -0.9], [0.0, -0.6], [0.0, -0.3], [0.0, 0.05], [0.0, 0.3], [0.0, 0.6], [0.0, 0.9],
        [0.2, -0.9], [0.2, -0.6], [0.2, -0.3], [0.2, 0], [0.2, 0.3], [0.2, 0.6], [0.2, 0.9],
        [0.4, -0.9], [0.4, -0.6], [0.4, -0.3], [0.4, 0], [0.4, 0.3], [0.4, 0.6], [0.4, 0.9],
        [0.6, -0.9], [0.6, -0.6], [0.6, -0.3], [0.6, 0], [0.6, 0.3], [0.6, 0.6], [0.6, 0.9]]
	def get_states(self, save_img=None):    
	    states, images_last, min_dists, collisions, scans, vels = self.get_robots_state()
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
	
	def odom_thread(self):
	    while True:
	        #print("***in send odometry thread***")
	        self.lv, self.av, robot_rotation = self.test_rob.get_current_vel()
	        print("in get states lv is", self.lv)
	        print("in get states av is", self.av)
	        print("in get states angle is ", robot_rotation)
	        print("in get states angle is ", robot_rotation[2])
	        self.send_data.send_odom(self.lv[0], self.av[2], robot_rotation[2]* 180 / math.pi)
	        #time.sleep(0.01)
	        
        
	def get_robots_state(self):
	    state = self.state_last
	    image = self.image_trans(state.laser_image)
	    goal_pose = [state.pose.position.x*self.meters_per_unit, state.pose.position.y*self.meters_per_unit]
	    #goal_pose = [state.pose.position.x, state.pose.position.y]
	    min_dist = state.min_dist.point.z
	    is_collision = state.collision
	    scan = state.laser
	    vel = state.vel
	    #print("*"*20)
	    #print("min_dist is ", min_dist)
	    #print("*"*20)
	    return goal_pose, image, min_dist, is_collision, scan, vel

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
	    print("reward is: ", reward)
	    if done < 0 and self.done == 0:
	        self.done = done
	    return (reward, done)

	def reset(self, target_dist = 0.0):
	    #self.robot_control([0, 0])
	    # random env
	    print("RANDOM ENVIRONMENT!!!!!!!!!!!!!!!!!!!!!!!!!!")
	    self.dr_interface.randomize_once()
	    time.sleep(0.5)
	    # random robot pose x y z, angle
	    stage = self.kit.get_stage()
	    self.kit.play()
	    while self.kit.is_loading():
	        self.kit.update(1 / 60.0)
	    #prim_path = self.omni.usd.get_stage_next_free_path(stage, self.prefix, False)
	    #print("prim_path is ", prim_path)
	    robot_prim = stage.GetPrimAtPath(self.prefix)
	    #print("robot_prim is ", robot_prim)
	    self.reset_robots(target_dist=target_dist)
	    #time.sleep(0.5)
	    print("in reset, initial pose is ", self.init_poses)
	    print("in reset, target pose is ", self.target_poses)
        # verify wheel speed
        # original 
	    #self.test_rob.teleport(robot_prim, (self.init_poses[0][0],self.init_poses[0][1],5), self.init_poses[0][2]* 180 / math.pi, self.kit)
        # toward +x direction
	    self.test_rob.teleport(robot_prim, (self.init_poses[0][0],self.init_poses[0][1],5),0, self.kit)
        # go +y direction 
	    #self.test_rob.teleport(robot_prim, (self.init_poses[0][0],self.init_poses[0][1],5),90, self.kit)
	    #self.test_rob.teleport(robot_prim, (self.init_poses[0][0],self.init_poses[0][1],5),45, self.kit)
	    #self.test_rob.teleport(robot_prim, (self.init_poses[0][0],self.init_poses[0][1],5),270, self.kit)
	    #self.robot_control([0, 0])
	    self.kit.update(1 / 60.0)
	    time.sleep(0.5)
	    self.last_d_obs = -1
	    self.last_d = -1
	    self.done = 0
	    self.reset_count += 1
	    self.step_count = 0
	    return self.get_states()

	def reset_robots(self, target_dist=0.0):
	    self.init_poses = []
	    self.target_poses = []
	    #start
	    while True:
	        pose_range = self.envs_cfg['begin_poses'][0]
	        rand_pose = self.random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
	        if self.free_check_robot(rand_pose[0], rand_pose[1], self.init_poses):
	            self.init_poses.append(rand_pose[:])
	            break
	    #goal
	    while True:
	        pose_range = self.envs_cfg['target_poses'][0]
	        rand_pose = self.random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
	        if (self.init_poses[0][0] - rand_pose[0]) ** 2 + (self.init_poses[0][1] - rand_pose[1]) ** 2 > target_dist ** 2 and self.free_check_robot(rand_pose[0], rand_pose[1], self.target_poses):
	            self.target_poses.append(rand_pose[:])
	            break
	    print("init poses are: ", self.init_poses)
	    print("target poses are: ", self.target_poses)
	    #self.init_poses = [[element / self.meters_per_unit for element in self.init_poses[0]]]
	    #self.target_poses = [[element / self.meters_per_unit for element in self.target_poses[0]]]
	    self.init_poses = [[self.init_poses[0][0] / self.meters_per_unit, self.init_poses[0][1] / self.meters_per_unit, self.init_poses[0][2]]]
	    self.target_poses = [[self.target_poses[0][0] / self.meters_per_unit, self.target_poses[0][1] / self.meters_per_unit, self.target_poses[0][2]]]

	    #print("init poses in omniverse are: ", self.init_poses)
	    #print("target poses in omniverse are: ", self.target_poses)
	    self.publish_goal(self.target_poses)

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
	def step_discrete(self, action):
	    return self.step(self.discrete_actions[action])

	def step(self, action):
	    #print("-"*50)
	    print("action in step is ", action)
	    self.robot_control(action)
	    #rospy.sleep(self.control_hz)
	    self.step_count += 1
	    states = self.get_states()
	    self.kit.update(1 / 60.0)
	    print("step number ", self.step_count)
	    #print("states in this step is ", states)
	    #1 verify the received data
	    #print("goal_pose in this step is ", states[0])
	    #print("min_dist in this step is ", states[2])
	    #print("is_collision in this step is ", states[3])
	    rw = self.get_rewards(states[0], states[2], states[3])
	    
	    if rw == False:
	        return False, False, False
	    else:
	        return states, np.array(rw[0], dtype='float64'), np.array(rw[1])

	def robot_control(self, action):
<<<<<<< HEAD
	    # wheel_back_left, wheel_back_right, wheel_front_left, wheel_front_right
	    #self.converted_cmd = self.convert_speed(action)
	    #self.test_rob.command(self.converted_cmd)
	    self.test_rob.command(action)
	    #self.test_rob.command((-20, 20, -20, 20))
	    #self.test_rob.commands(action, self.init_poses[0][2])
	    self.kit.update(1 / 60.0)
	    print("done is ",self.done)
	    super().robot_control(action)

=======
            # wheel_back_left, wheel_back_right, wheel_front_left, wheel_front_right
            #self.converted_cmd = self.convert_speed(action)
            #self.test_rob.command(self.converted_cmd)
            ########################################################check
            action = [0.0, 0.9]
            self.test_rob.command(action)
            #self.test_rob.command((-20, 20, -20, 20))
            ########################################################check
            #self.test_rob.commands(action, self.init_poses[0][2])
            #self.test_rob.commands([0.6,0.6], self.init_poses[0][2])
            self.kit.update(1 / 60.0)
            vel = Twist()
            print("done is ",self.done)
            if self.done == 0:
                #vel.linear.x = action[0]/self.meters_per_unit
                #vel.angular.z = action[1]
                #vel.linear.x = self.lv[0]/self.meters_per_unit
                vel.linear.x = self.lv[0]
                vel.angular.z = self.av[2]
            else:
                vel.linear.x = 0
                vel.angular.z = 0
            #2 verify cmd_vel
            #print("linear x is ", vel.linear.x)
            #print("vel.angular z is ", vel.angular.z)
            self.vel_pub.publish(vel)
>>>>>>> 04003a291fa70405828bd741e61ec45cd91c36c3
	def convert_speed(self, action):
	    v_1 = v_2 = v_3 = v_4 = 0
	    robot_length = 0.73
	    robot_width = 0.495
	    if (action[0] + action[1]) != 0:
	        v_1 = -(action[0] - (robot_length + robot_width) * action[1])/self.meters_per_unit
	        v_2 = (action[0] + (robot_length + robot_width) * action[1])/self.meters_per_unit
	        v_3 = (action[0] + (robot_length + robot_width) * action[1])/self.meters_per_unit
	        v_4 = -(action[0] - (robot_length + robot_width) * action[1])/self.meters_per_unit
	    #print("back left speed ", v_4)
	    #print("back right speed ", v_3)
	    #print("front left speed ", v_1)
	    #print("front right speed ", v_2)

	    return (v_4, v_3, v_1, v_2)

	def env_init(self):
	    #from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
	    #result, nucleus = find_nucleus_server()
	    env_path = "omniverse://nucleus-01.lscm.ml/Library/Environments/test_env.usd"
	    #env_path = nucleus + "/Library/Environments/test_env.usd"
	    print(env_path)
	    self.omni.usd.get_context().open_stage(env_path, None)
	    obj_list = self.test_env.create_objects(0,0,0)
	    print(obj_list)
	    self.test_env.domain_randomization_test(obj_list)

	def robot_init(self):
	    stage = self.kit.get_stage()
	    print("stage is ", stage)
	    prim_path = self.omni.usd.get_stage_next_free_path(stage, self.prefix, False)
	    print("prim_path is ", prim_path)
	    robot_prim = stage.DefinePrim(prim_path, "Xform")
	    print("robot_prim is ", robot_prim)
	    robot_prim.GetReferences().AddReference("/Library/Robots/config_robot/robot_event_cam.usd")
	    self.test_rob.spawn(stage, robot_prim, prim_path)
	    ext_manager = self.omni.kit.app.get_app().get_extension_manager()
	    ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)
	    self.kit.play()
	    while self.kit.is_loading():
	        self.kit.update(1 / 60.0)
	    robot_prim = stage.GetPrimAtPath(self.prefix)
	    self.reset_robots(0.0)
	    self.test_rob.teleport(robot_prim, (self.init_poses[0][0],self.init_poses[0][1],-10), self.init_poses[0][2]* 180 / math.pi, self.kit)
	    self.send_odom_thread = threading.Thread(target=self.odom_thread)
	    self.send_odom_thread.start()
        
	def init_datas(self):
	    self.last_d = -1
	    self.last_d_obs = -1
	    self.images_batch = [None] * self.image_batch
	    self.images_ptr = []
	    self.obs_name = []
	    self.done = 0
	    self.obstacles_ranges = []
	    self.images_ptr.append(0)
	    self.obstacles_ranges.append([])
