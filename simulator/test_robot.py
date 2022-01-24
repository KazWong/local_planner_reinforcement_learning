import numpy as np
import math

class Robot_config:
    def __init__(self, stage, omni):
        from pxr import UsdGeom
        self.omni = omni
        from omni.isaac.dynamic_control import _dynamic_control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.omni = omni
        self.ar = None
        self.stage = stage
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.omni.usd.get_context().get_stage())
        self.lx_ly = 0.43

    def spawn(self, stage, robot_prim, prim_path):
        from pxr import UsdGeom, Gf, UsdPhysics
        TRANSLATION_RANGE = 1000.0
        translation = np.random.rand(3) * TRANSLATION_RANGE
        angle = np.random.rand(1)
        print(robot_prim)
        xform = UsdGeom.Xformable(robot_prim)
        print(xform)
        print("stage is ", stage)
        print("prim path is ", prim_path)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(translation.tolist())
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), (angle[0])))
        xform_op.Set(mat)
        DRIVE_STIFFNESS = 10000.0
        #DRIVE_STIFFNESS = 0
        # Set joint drive parameters
        wheel_back_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_left_joint"), "angular")
        wheel_back_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)

        wheel_back_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_right_joint"), "angular")
        wheel_back_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)

        wheel_front_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_left_joint"), "angular")
        wheel_front_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)

        wheel_front_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_right_joint"), "angular")
        wheel_front_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        self.ar = self.dc.get_articulation(robot_prim.GetPath().pathString)
        
    def teleport(self, robot_prim, location, rotation, kit, settle=True):
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
        self.command([0,0])
        # Settle the robot onto the ground
        if settle:
            frame = 0
            velocity = 1
            while velocity > 0.1 and frame < 120:
                kit.update(1.0 / 60.0)
                lin_vel = self.dc.get_rigid_body_linear_velocity(chassis)
                velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
                frame = frame + 1

    def IK(self, cmd_vel):
        v = [0, 0, 0, 0]
        v[0] = - cmd_vel[0] + cmd_vel[1] + (self.lx_ly)*cmd_vel[2];
        v[1] =   cmd_vel[0] + cmd_vel[1] + (self.lx_ly)*cmd_vel[2];
        v[2] = - cmd_vel[0] - cmd_vel[1] + (self.lx_ly)*cmd_vel[2];
        v[3] =   cmd_vel[0] - cmd_vel[1] + (self.lx_ly)*cmd_vel[2];
        return v

    def FK(self, feedback):
        odom = [0, 0, 0]
        odom[0] = (-feedback[0] + feedback[1] - feedback[2] + feedback[3]) / 4.
        odom[1] = ( feedback[0] + feedback[1] - feedback[2] - feedback[3]) / 4.
        odom[2] = ( feedback[0] + feedback[1] + feedback[2] + feedback[3]) / (self.lx_ly*4.)
        return odom

    def command(self, action):
        #print("linear speed is ", action[0], " ", action[0]/self.meters_per_unit)
        #print("angular speed is ", action[1])
        cmd_vel = [action[0]/self.meters_per_unit, 0.0, action[1]]
        #cmd_vel = [0.5/self.meters_per_unit, 0.0, 0.0]
        chassis = self.dc.get_articulation_root_body(self.ar)
        #num_joints = self.dc.get_articulation_joint_count(self.ar)
        #num_dofs = self.dc.get_articulation_dof_count(self.ar)
        #num_bodies = self.dc.get_articulation_body_count(self.ar)

        wheel_back_left = self.dc.find_articulation_dof(self.ar, "wheel_back_left_joint")
        wheel_back_right = self.dc.find_articulation_dof(self.ar, "wheel_back_right_joint")
        wheel_front_left = self.dc.find_articulation_dof(self.ar, "wheel_front_left_joint")
        wheel_front_right = self.dc.find_articulation_dof(self.ar, "wheel_front_right_joint")

        self.dc.wake_up_articulation(self.ar)

        motor_value = self.IK(cmd_vel)
        wheel_back_left_speed = self.wheel_speed_from_motor_value(motor_value[0])
        wheel_back_right_speed = self.wheel_speed_from_motor_value(motor_value[1])
        wheel_front_left_speed = self.wheel_speed_from_motor_value(motor_value[2])
        wheel_front_right_speed = self.wheel_speed_from_motor_value(motor_value[3])

        #print("wheel_back_left_speed is ", wheel_back_left_speed)
        #print("wheel_back_right_speed is ", wheel_back_right_speed)
        #print("wheel_front_left_speed is ", wheel_front_left_speed)
        #print("wheel_front_right_speed is ", wheel_front_right_speed)
        
        self.dc.set_dof_velocity_target(wheel_back_left, np.clip(wheel_back_left_speed, -100, 100))
        self.dc.set_dof_velocity_target(wheel_back_right, np.clip(wheel_back_right_speed, -100, 100))
        self.dc.set_dof_velocity_target(wheel_front_left, np.clip(wheel_front_left_speed, -100, 100))
        self.dc.set_dof_velocity_target(wheel_front_right, np.clip(wheel_front_right_speed, -100, 100))

    def commands(self,action, angle):
        #print("linear speed is", action[0]/self.meters_per_unit)
        #print("angular speed is", action[1]/self.meters_per_unit)
        chassis = self.dc.get_articulation_root_body(self.ar)
        self.dc.wake_up_articulation(self.ar)
        #print("radian is ", angle)
        #print("degree is ", angle * 180 / math.pi)
        #print("x speed is ", action[0] / self.meters_per_unit * math.cos(angle * 180 / math.pi))
        #print("y speed is ", -action[0] / self.meters_per_unit * math.sin(angle * 180 / math.pi))
        #self.dc.set_rigid_body_linear_velocity(chassis, [action[0] / self.meters_per_unit * math.cos(angle * 180 / math.pi), -action[0] / self.meters_per_unit * math.sin(angle * 180 / math.pi), 0])
        #self.dc.set_rigid_body_angular_velocity(chassis, [0, 0, action[1]])
        print("in commands action is ", action[0])
        print("in commands action is ", action[1])       
        print("in commands action ros is ", action[0]/ self.meters_per_unit)        
        self.dc.set_rigid_body_linear_velocity(chassis, [action[0] / self.meters_per_unit, 0, 0])
        self.dc.set_rigid_body_angular_velocity(chassis, [0, 0, action[1]])
        #self.dc.set_rigid_body_linear_velocity(chassis, [100, -100, 0])
        #self.dc.set_rigid_body_angular_velocity(chassis, [0, 0, -10])

    # idealized motor model
    def wheel_speed_from_motor_value(self, motor_input):
        #print("speed is ",motor_input)
        return motor_input

    def get_lidar_data(self):
        from omni.isaac.range_sensor import _range_sensor
        lidarInterface = _range_sensor.acquire_lidar_sensor_interface()

        depth = lidarInterface.get_depth_data("/World/soap_odom/agv_lidar/Lidar")
        zenith = lidarInterface.get_zenith_data("/World/soap_odom/agv_lidar/Lidar")
        azimuth = lidarInterface.get_azimuth_data("/World/soap_odom/agv_lidar/Lidar")

        #print("depth", depth/65535*100)
        #print("zenith", zenith)
        #print("azimuth", azimuth)
        return [depth/65535*100, azimuth]

    def get_current_vel(self):
        chassis = self.dc.get_articulation_root_body(self.ar)
        linear_vel = self.dc.get_rigid_body_linear_velocity(chassis)
        #local_linear_vel = self.dc.get_rigid_body_local_linear_velocity(chassis)
        #norm_linear_vel = np.linalg.norm([linear_vel.x, linear_vel.y, linear_vel.z])
        #norm_local_linear_vel = np.linalg.norm([local_linear_vel.x, local_linear_vel.y, local_linear_vel.z])
        angular_vel = self.dc.get_rigid_body_angular_velocity(chassis)
        current_pose = self.dc.get_rigid_body_pose(chassis)
        #linear_vel = (linear_vel[0]*self.meters_per_unit, linear_vel[1]*self.meters_per_unit, linear_vel[2]*self.meters_per_unit)
        print("current world position is ", current_pose.p)
        print("rotation is ", current_pose.r)
        print("linear vel in get current vel is ", (linear_vel[0]*self.meters_per_unit, 0.0,0.0))
        #print("local linear vel in get current vel is ", (local_linear_vel[0]*self.meters_per_unit, local_linear_vel[1]*self.meters_per_unit, local_linear_vel[2]*self.meters_per_unit))
        #print("norm linear vel in get current vel is ", norm_linear_vel)
        #print("norm local linear vel in get current vel is ", norm_local_linear_vel)
        print("angular vel is ", (0.0, 0.0, angular_vel[2]))
        #linear_vel = [linear_vel[0]*self.meters_per_unit * math.cos(angular_vel[2] * 180 / math.pi), -linear_vel[0]*self.meters_per_unit * math.sin(angular_vel[2] * 180 / math.pi), 0]
        #angular_vel = [angular_vel[0],angular_vel[1],angular_vel[2]]
        #print("final linear vel is ", (linear_vel[0]*self.meters_per_unit * math.cos(angular_vel[2] * 180 / math.pi), -linear_vel[0]*self.meters_per_unit * math.sin(angular_vel[2] * 180 / math.pi), 0))
        #linear_vel = [linear_vel[0]*self.meters_per_unit, linear_vel[1]*self.meters_per_unit, 0.0]
        #angular_vel = [angular_vel[0],angular_vel[1],angular_vel[2]]
        linear_vel = [linear_vel[0]*self.meters_per_unit, 0.0, 0.0]
        angular_vel = [0.0, 0.0,angular_vel[2]]        
        return linear_vel, angular_vel, current_pose.r

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
