import rosparam
import os
from tf.transformations import euler_from_quaternion
import numpy as np

X=0
Y=1
YAW=2

def LoadYamlFileParams(path_config_file):
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)

def pose_to_euler(pose):
    _, _, yaw = euler_from_quaternion([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])
    yaw = np.mod(yaw, 2*np.pi)
    return [pose.position.x, pose.position.y, yaw]

def signed_yaw_diff(p1, p2):
    return np.mod((p2[YAW] - p1[YAW]) + np.pi, 2*np.pi) - np.pi

def get_diff(p1, p2):
    if isinstance(p1, list):
        p1 = np.array(p1)
    if isinstance(p2, list):
        p2 = np.array(p2)
    diff = p2-p1
    diff[YAW] = signed_yaw_diff(p1, p2)
    return diff

def get_distance(p1, p2):
    diff = get_diff(p1, p2)
    return np.sqrt(np.sum(np.power(np.asarray(diff), 2)))

def get_relative_position(absolute_pose, absolute_position):
    # pose is robot
    # position is goal
    M = np.array(
        [
        [np.cos(absolute_pose[YAW]), np.sin(absolute_pose[YAW])], 
        [-np.sin(absolute_pose[YAW]), np.cos(absolute_pose[YAW])]
        ]
    )
    X_position = np.array(
        [
        [absolute_position[X]],
        [absolute_position[Y]]
        ]
    )
    X_pose = np.array(
        [
        [absolute_pose[X]],
        [absolute_pose[Y]]
        ]
    )

    relative_position = np.matmul(M,(X_position - X_pose))
    relative_position = np.reshape(relative_position, (2,))

    return relative_position