import ros_utils as util
import os.path as osp
import rospy
from gazebo_msgs.srv import *
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist

def delete_model(model_name):
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp = delete_model_srv(model_name)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def spawn_model(model_name, model_xml, initial_pose=[0,0], robot_namespace='', reference_frame='world'):
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp = spawn_sdf_model(model_name, model_xml, robot_namespace, util.to_pose_msg(initial_pose), reference_frame)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def get_model_sdf(model_name):
    pkg_path = util.get_pkg_path('gz_pkg')
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
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def set_model_state(model_name, pose, twist=None, frame='world'):
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_model = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        if twist == None:
            tw = Twist()
        model_state = ModelState()
        model_state.model_name = model_name
        model_state.pose = util.to_pose_msg(pose)
        model_state.twist = tw
        model_state.reference_frame = frame
        resp = set_model(model_state)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
