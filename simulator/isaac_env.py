from env import Env
import numpy as np
import random
import math
import os
import sys
import signal
import argparse
from argparse import Namespace
from test_env import Env_config
from test_robot import Robot_config
from omni.isaac.python_app import OmniKitHelper

class IsaacEnv(Env):
	def __init__(self, cfg_names):
	    super().__init__(cfg_names)
	    self.startup_config = {"renderer": "RayTracedLighting", "headless": False, "experience": '/home/slam/.local/share/ov/pkg/isaac_sim-2021.1.1/apps/omni.isaac.sim.python.kit'}
	    self.kit = OmniKitHelper(self.startup_config)
	    import omni
	    self.omni = omni
	    self.test_env = Env_config(omni,self.kit)
	    self.stage = self.kit.get_stage()
	    self.test_rob = Robot_config(self.stage, self.omni)
	    import omni.isaac.dr as dr
	    self.dr_interface = dr._dr.acquire_dr_interface()
	    self.prefix = "/World/" + self.robot_name
	    self.pose_differ = []
	    self.scan = []
	    self.min_dist = 999
	    image_width_ = 60
	    image_height_ = 60
	    resolution_ = 0.1
	    width = image_width_ * resolution_;
	    height = image_height_ * resolution_;
	    self.max_dis_ = math.sqrt(width * width / 4.0 + height * height / 4.0);
	    self.angle_increment = 0.4
	    ###tf_laser_base_ =  tobeedited
	    self.discrete_actions = \
        [[0.0, -0.9], [0.0, -0.6], [0.0, -0.3], [0.0, 0.05], [0.0, 0.3], [0.0, 0.6], [0.0, 0.9],
        [0.2, -0.9], [0.2, -0.6], [0.2, -0.3], [0.2, 0], [0.2, 0.3], [0.2, 0.6], [0.2, 0.9],
        [0.4, -0.9], [0.4, -0.6], [0.4, -0.3], [0.4, 0], [0.4, 0.3], [0.4, 0.6], [0.4, 0.9],
        [0.6, -0.9], [0.6, -0.6], [0.6, -0.3], [0.6, 0], [0.6, 0.3], [0.6, 0.6], [0.6, 0.9]]
	    
	def get_states(self):
	    #states, images_last, min_dists, collisions, scans, vels = self.get_robots_state()
	    states, images_last, min_dists, collisions, scans, vels = self.get_robots_state_isaac()
	    return
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
	    
	def get_robots_state(self):
	    state = self.state_last # from ros sub msg
	    image = self.image_trans(state.laser_image)
	    goal_pose = [state.pose.position.x, state.pose.position.y]
	    min_dist = state.min_dist.point.z
	    is_collision = state.collision
	    scan = state.laser
	    vel = state.vel
	    return goal_pose, image, min_dist, is_collision, scan, vel
                
	def get_robots_state_isaac(self):
	    #state = self.state_last # from ros sub msg
	    #image = self.image_trans(state.laser_image)
	    #goal_pose = [state.pose.position.x, state.pose.position.y]
	    #min_dist = state.min_dist.point.z
	    #is_collision = self.test_rob.check_overlap_box()
	    #self.test_rob.check_overlap_box()
	    #scan = state.laser
	    #vel = state.vel
	    # edited
	    state = None
	    # TODO draw data from scan to image
	    image = None
	    is_collision = self.test_rob.check_overlap_box()
	    self.scan = self.test_rob.get_lidar_data()
	    vel = self.test_rob.get_current_vel()
	    goal_pose = self.pose_differ
	    # TODO fix min_dist
	    self.min_dist = self.min_dist_cal(self.scan)
	    return goal_pose, image, self.min_dist, is_collision, self.scan, vel
	    
	def get_rewards(self):
	    pass
	
	def reset(self, target_dist):
	    # random env
	    self.dr_interface.randomize_once()
	    # random robot pose x y z, angle
	    stage = self.kit.get_stage()
	    self.kit.play()
	    print("reset")
	    #prim_path = self.omni.usd.get_stage_next_free_path(stage, self.prefix, False)
	    #print("prim_path is ", prim_path)
	    robot_prim = stage.GetPrimAtPath(self.prefix)
	    print("robot_prim is ", robot_prim)
	    self.reset_robots(target_dist=target_dist)
	    print("in reset, initial pose is ", self.init_poses)
	    print("in reset, target pose is ", self.target_poses)
	    
	    self.test_rob.teleport(robot_prim, (self.init_poses[0][0],self.init_poses[0][1],30), self.init_poses[0][2])
	    
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
	    #self.publish_goal(self.target_poses)
	    self.pose_differ = [self.target_poses[0][0] - self.init_poses[0][0], self.target_poses[0][1] - self.init_poses[0][1]]
	    
	    
	def random_pose(self, x, y, sita):
	    return [random.uniform(x[0],x[1]), random.uniform(y[0],y[1]), random.uniform(sita[0],sita[1])]
	
	def min_dist_cal(self, scan):
	    #print("scan distance to calculate min dist is ", scan[0][0])
	    #print("scan angle to calculate min dist is ", scan[1][0])
	    #print("scan distance to calculate min dist is ", scan[0][-1])
	    #print("scan angle to calculate min dist is ", scan[1][-1])
	    #print("scan is ", scan)
	    d = 0
	    for i in range(len(scan[0])):
	        if scan[0][i][0] == 100:
	            d = self.max_dis_
	        else:
	            d = scan[0][i][0]
	        theta = scan[1][0] + self.angle_increment * i
	        if theta < scan[1][0] or theta > scan[1][-1]:
	            continue
	        x = d * math.cos(theta);
	        y = d * math.sin(theta);
	        xy = (x,y,0)
	        #xy_base = self.tf_laser_base_ * xy
	        print("xy is ", xy)
	        #print("xy_base is ", xy_base)

        #tf::Vector3 xy, xy_base;
        #xy.setValue(x, y, 0);
        #xy_base = tf_laser_base_ * xy;
        #double x_base = xy_base.getX();
        #double y_base = xy_base.getY();
        
        #//penalty for min distance
        #double dist = sqrt(x_base * x_base + y_base * y_base);
        #if (dist < state_msg.min_dist.point.z)
        #{
        #    state_msg.min_dist.point.x = x_base;
        #    state_msg.min_dist.point.y = y_base;
        #    state_msg.min_dist.point.z = dist;
        #}
	    return 0
	    
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
	    pass
	
	def robot_control(self, action):
	    # wheel_back_left, wheel_back_right, wheel_front_left, wheel_front_right
	    self.test_rob.command((-20, 20, -20, 20))
	    
	def env_init(self):
	    from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
	    result, nucleus = find_nucleus_server()
	    env_path = nucleus + "/Library/Environments/test_env.usd"
	    print(env_path)
	    self.omni.usd.get_context().open_stage(env_path, None)
	    obj_list = self.test_env.create_objects(3,3,0)
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
	
	def step_discrete(self, action):
	    pass
