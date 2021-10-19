import tf
from geometry_msgs.msg import Pose, Quaternion, Point
import rospkg
import math
import numpy as np

def get_pkg_path(pkg_name):
    rospack = rospkg.RosPack()
    return rospack.get_path(pkg_name)

def q_to_rpy(q):
    if type(q) == list or type(q) == tuple:
        quaternion = q
    elif type(q) == Pose:
        quaternion = (q.orientation.x, q.orientation.y,
                      q.orientation.z, q.orientation.w)
    elif type(q) == Quaternion:
        quaternion = (q.x, q.y, q.z, q.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    return (roll, pitch, yaw)

def rpy_to_q(rpy):
    return tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

def matrix_from_t_q(t, q):
    return tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(t), tf.transformations.quaternion_matrix(q))

def matrix_from_t_m(t, m):
    return np.array([[m[0][0], m[0][1], m[0][2], t[0]],
                     [m[1][0], m[1][1], m[1][2], t[1]],
                     [m[2][0], m[2][1], m[2][2], t[2]],
                     [0,       0,       0,       1]
                    ])

def matrix_from_pose(pose):
    if type(pose) == list or type(pose) == tuple:
        return matrix_from_t_q(pose[:3], pose[3:])
    elif type(pose) == Pose:
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        position = (pose.position.x, pose.position.y, pose.position.z)
        return matrix_from_t_q(position, quaternion)

def q_from_matrix(m):
    return tf.transformations.quaternion_from_matrix(m)

def t_from_matrix(m):
    return tf.transformations.translation_from_matrix(m)

def rpy_from_matrix(m):
    return tf.transformations.euler_from_matrix(m)

def inverse(m):
    return tf.transformations.inverse_matrix(m)

def transform_point(m, point):
    xyz = tuple(np.dot(m, np.array([point[0], point[1], point[2], 1.0])))[:3]
    return xyz

def mul_matrix(m1, m2):
    return np.dot(m1, m2)

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
            q = rpy_to_q(data[3:])
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
            q = rpy_to_q([0, 0, data[2]])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            return pose
        else:
            return False
