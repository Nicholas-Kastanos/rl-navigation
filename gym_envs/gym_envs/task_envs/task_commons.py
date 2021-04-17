import rosparam
import os
from tf.transformations import euler_from_quaternion
import numpy as np

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
    return [pose.position.x, pose.position.y, yaw]

def get_distance(p1, p2):
    if isinstance(p1, list):
        p1 = np.array(p1)
    if isinstance(p2, list):
        p2 = np.array(p2)
    return np.linalg.norm(p2-p1)
        