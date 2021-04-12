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

    yaw = np.mod(yaw, 2*np.pi)
    return [pose.position.x, pose.position.y, yaw]
        