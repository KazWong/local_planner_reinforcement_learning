import numpy as np
import random
import os
import sys
import signal
import argparse
from argparse import Namespace

from omni.isaac.python_app import OmniKitHelper

class Robo(object):
	def __init__(self, robot_name, omni, nucleus):
		self.omni = omni
		self.nucleus = nucleus
		self.path = "/Library/Robots/"
		self.name = robot_name

	def spawn(self):
		prefix = "/World/" + robot_name
		prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
		print(prim_path)
		robot_prim = stage.DefinePrim(prim_path, "Xform")
		robot_prim.GetReferences().AddReference(self.path + self.name)
		self.xform = UsdGeom.Xformable(robot_prim)
		self.xform_op = self.xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")

		omni.kit.commands.execute(
	        "ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/odom/robot/agv_lidar/ROS_Lidar.enabled"), value=True, prev=None)
		omni.kit.commands.execute(
	        "ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/odom/robot/ROS_PoseTree.enabled"), value=True, prev=None)
		omni.kit.commands.execute(
	        "ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/odom/robot/ROS_JointState.enabled"), value=True, prev=None)
		omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/ROS_Clock.enabled"), value=True, prev=None)

	def teleport(self):
		translation = np.random.rand(3) * TRANSLATION_RANGE
		angle = np.random.rand(1)

		mat = Gf.Matrix4d().SetTranslate(translation.tolist())
		mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), (angle[0])))
		self.xform_op.Set(mat)

	def cmd_vel(self, action):
		
