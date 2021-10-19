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
        self.cfg_names = cfg_names
        self.startup_config = {
		"renderer": "RayTracedLighting",
		"headless": args.headless,
		"experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
	    }
	    self.kit = OmniKitHelper(self.startup_config)
	    import omni
	    self.test_env = Env_config(omni,self.kit)
	    self.stage = self.kit.get_stage()
	    self.prefix = "/World/soap_odom"
        self.prim_path = omni.usd.get_stage_next_free_path(self.stage, self.prefix, False)
	    self.robot_prim = self.stage.DefinePrim(self.prim_path, "Xform")
	    self.test_rob = Robot_config(self.stage, omni, self.robot_prim)
	    
    def get_states(self):
        pass

    def get_rewards(self):
        pass

    def reset(self):
        # random env
        dr_interface.randomize_once()
        # random robot pose x y z, angle
        self.test_rob.teleport((0,0,30), 0)
        
    def step_discrete(self, action):
        pass

    def robot_control(self, action):
        # wheel_back_left, wheel_back_right, wheel_front_left, wheel_front_right
        self.test_rob.command((-20, 20, -20, 20))
        
    def env_init(self):
        import omni.isaac.dr as dr
        obj_list = self.test_env.create_objects(4,4,4)
        dr_interface = dr._dr.acquire_dr_interface()
        self.test_env.domain_randomization_test(obj_list)

    def robot_init(self):
        TRANSLATION_RANGE = 1000.0
        translation = np.random.rand(3) * TRANSLATION_RANGE
        angle = np.random.rand(1)
        
        self.robot_prim.GetReferences().AddReference(args.robo_path)
        xform = UsdGeom.Xformable(robot_prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(translation.tolist())
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), (angle[0])))
        xform_op.Set(mat)
        DRIVE_STIFFNESS = 10000.0
        # Set joint drive parameters
        wheel_back_left_joint = UsdPhysics.DriveAPI.Apply(self.stage.GetPrimAtPath(f"{self.prim_path}/agv_base_link/wheel_back_left_joint"), "angular")
        wheel_back_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_back_right_joint = UsdPhysics.DriveAPI.Apply(self.stage.GetPrimAtPath(f"{self.prim_path}/agv_base_link/wheel_back_right_joint"), "angular")
        wheel_back_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_front_left_joint = UsdPhysics.DriveAPI.Apply(self.stage.GetPrimAtPath(f"{self.prim_path}/agv_base_link/wheel_front_left_joint"), "angular")
        wheel_front_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_front_right_joint = UsdPhysics.DriveAPI.Apply(self.stage.GetPrimAtPath(f"{self.prim_path}/agv_base_link/wheel_front_right_joint"), "angular")
        wheel_front_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
