import rospy
from gazebo_msgs.srv import *
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
import ros_utils as util
import rospkg
import os.path as osp

def delete_model(model_name):
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp = delete_model_srv(model_name)
        return resp.success
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)

def spawn_model(model_name, model_xml, initial_pose=[0,0], robot_namespace='', reference_frame='world'):
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp = spawn_sdf_model(model_name, model_xml, robot_namespace, to_pose_msg(initial_pose), reference_frame)
        return resp.success
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)

def to_pose_msg(data):
    pose = Pose()
    if type(data) == list or type(data) == tuple:
        if len(data) == 2:
            pose.position.x = data[0]
            pose.position.y = data[1]
            pose.orientation.w = 1
            return pose
        pose.position.x = data[0]
        pose.position.y = data[1]
        if len(data) == 6:
            pose.position.z = data[2]
            q = util.rpy_to_q(data[3:])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            return pose
        elif len(data) == 7:
            pose.position.z = data[2]
            pose.orientation.x = data[3]
            pose.orientation.y = data[4]
            pose.orientation.z = data[5]
            pose.orientation.w = data[6]
            return pose
        elif len(data) == 3:
            pose.position.z = 0
            q = util.rpy_to_q([0, 0, data[2]])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            return pose
        else:
            return False    

def get_pkg_path(pkg_name):
    rospack = rospkg.RosPack()
    return rospack.get_path(pkg_name)   

def get_model_sdf(model_name):
    pkg_path = get_pkg_path('gz_pkg')
    final_file = osp.join(pkg_path, 'sdf', model_name, 'model.sdf')
    with open(final_file,'r') as f:
        sdf = f.read()
        return sdf

def get_world_models():
    rospy.wait_for_service('/gazebo/get_world_properties')
    try:
        get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        resp = get_world_properties()
        return resp.model_names
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)

def set_model_state(model_name, pose, twist=None, frame='world'):
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_model = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        if twist == None:
            tw = Twist()
        model_state = ModelState()
        model_state.model_name = model_name
        model_state.pose = to_pose_msg(pose)
        model_state.twist = tw
        model_state.reference_frame = frame
        resp = set_model(model_state)
        return resp.success
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)
