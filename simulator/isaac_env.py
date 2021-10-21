from env import Env
import numpy as np
import random
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
	    
	def get_states(self):
	    pass
	
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
	    self.test_rob.teleport(robot_prim, (0,0,30), 0)
	
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
