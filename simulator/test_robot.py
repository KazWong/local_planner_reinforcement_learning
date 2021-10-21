import numpy as np

class Robot_config:
    def __init__(self, stage, omni):
        self.omni = omni
        from omni.isaac.dynamic_control import _dynamic_control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.omni = omni
        self.ar = None
        self.stage = stage

    def spawn(self, stage, robot_prim, prim_path):
        from pxr import UsdGeom, Gf, UsdPhysics
        TRANSLATION_RANGE = 1000.0
        translation = np.random.rand(3) * TRANSLATION_RANGE
        angle = np.random.rand(1)
        print("X" * 10)
        print(robot_prim)
        xform = UsdGeom.Xformable(robot_prim)
        print(xform)
        print("stage is ", stage)
        print("prim path is ", prim_path)
        print("X"* 10)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(translation.tolist())
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), (angle[0])))
        xform_op.Set(mat)
        DRIVE_STIFFNESS = 10000.0
        # Set joint drive parameters
        wheel_back_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_left_joint"), "angular")
        wheel_back_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_back_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_right_joint"), "angular")
        wheel_back_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_front_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_left_joint"), "angular")
        wheel_front_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_front_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_right_joint"), "angular")
        wheel_front_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
	
    def teleport(self, robot_prim, location, rotation, settle=False):
        from pxr import Gf
        from omni.isaac.dynamic_control import _dynamic_control
        print("before teleport", self.ar)
        #if self.ar is None:
        print(type(robot_prim.GetPath().pathString), robot_prim.GetPath().pathString)
        self.ar = self.dc.get_articulation(robot_prim.GetPath().pathString)
        print("after teleport", self.ar)
        chassis = self.dc.get_articulation_root_body(self.ar)
    
        self.dc.wake_up_articulation(self.ar)
        rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation).GetQuaternion()
    
        tf = _dynamic_control.Transform(
            location,
            (rot_quat.GetImaginary()[0], rot_quat.GetImaginary()[1], rot_quat.GetImaginary()[2], rot_quat.GetReal()),
        )
        self.dc.set_rigid_body_pose(chassis, tf)
        self.dc.set_rigid_body_linear_velocity(chassis, [0, 0, 0])
        self.dc.set_rigid_body_angular_velocity(chassis, [0, 0, 0])
        #self.command((-20, 20, -20, 20))
        # Settle the robot onto the ground
        if settle:
            frame = 0
            velocity = 1
            while velocity > 0.1 and frame < 120:
                self.omni.usd.get_context().update(1.0 / 60.0)
                lin_vel = self.dc.get_rigid_body_linear_velocity(chassis)
                velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
                frame = frame + 1
    
    def command(self,motor_value): 
        chassis = self.dc.get_articulation_root_body(self.ar)
        #num_joints = self.dc.get_articulation_joint_count(self.ar)
        #num_dofs = self.dc.get_articulation_dof_count(self.ar)
        #num_bodies = self.dc.get_articulation_body_count(self.ar)
                           
        wheel_back_left = self.dc.find_articulation_dof(self.ar, "wheel_back_left_joint")
        wheel_back_right = self.dc.find_articulation_dof(self.ar, "wheel_back_right_joint")
        wheel_front_left = self.dc.find_articulation_dof(self.ar, "wheel_front_left_joint")
        wheel_front_right = self.dc.find_articulation_dof(self.ar, "wheel_front_right_joint")

        self.dc.wake_up_articulation(self.ar)

        wheel_back_left_speed = self.wheel_speed_from_motor_value(motor_value[0])
        wheel_back_right_speed = self.wheel_speed_from_motor_value(motor_value[1])
        wheel_front_left_speed = self.wheel_speed_from_motor_value(motor_value[2])
        wheel_front_right_speed = self.wheel_speed_from_motor_value(motor_value[3])
        
        self.dc.set_dof_velocity_target(wheel_back_left, np.clip(wheel_back_left_speed, -10, 10))
        self.dc.set_dof_velocity_target(wheel_back_right, np.clip(wheel_back_right_speed, -10, 10))      
        self.dc.set_dof_velocity_target(wheel_front_left, np.clip(wheel_front_left_speed, -10, 10))
        self.dc.set_dof_velocity_target(wheel_front_right, np.clip(wheel_front_right_speed, -10, 10))
    
    # idealized motor model
    def wheel_speed_from_motor_value(self, motor_input):
        print("speed is ",motor_input)
        return motor_input
        
    def check_overlap_box(self):
        # Defines a cubic region to check overlap with
        import omni.physx
        from omni.physx import get_physx_scene_query_interface
        import carb
        #print("*"*50)
        chassis = self.dc.get_articulation_root_body(self.ar)
        robot_base_pose = self.dc.get_rigid_body_pose(chassis)
        #print("chassis is ", chassis)
        #print("pose is ", robot_base_pose)
        print("pose is ", robot_base_pose.p)
        #print("*"*50)        
        extent = carb.Float3(38.0, 26.0, 5.0)
        # origin = carb.Float3(0.0, 0.0, 0.0)
        origin = robot_base_pose.p
        rotation = carb.Float4(0.0, 0.0, 1.0, 0.0)
        # physX query to detect number of hits for a cubic region
        numHits = get_physx_scene_query_interface().overlap_box(extent, origin, rotation, self.report_hit, False)
        print("num of overlaps ", numHits)
        # physX query to detect number of hits for a spherical region
        # numHits = get_physx_scene_query_interface().overlap_sphere(radius, origin, self.report_hit, False)
        #self.kit.update()
        return numHits > 1

    def report_hit(self, hit):
        from pxr import UsdGeom, Gf, Vt

        # When a collision is detected, the object colour changes to red.
    #    hitColor = Vt.Vec3fArray([Gf.Vec3f(180.0 / 255.0, 16.0 / 255.0, 0.0)])
    #    usdGeom = UsdGeom.Mesh.Get(self.stage, hit.rigid_body)
    #    usdGeom.GetDisplayColorAttr().Set(hitColor)
        return True
