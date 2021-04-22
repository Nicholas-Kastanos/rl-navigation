import os

import numpy as np
from geometry_msgs.msg import Pose, Quaternion

X=0
Y=1
YAW=2

def euler_angles_to_quaternion(roll=0, pitch=0, yaw=0):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w

def quaternion_to_euler_angles(x=0, y=0, z=0, w=0):
    if isinstance(x, list):
        return quaternion_to_euler_angles(x[0], x[1], x[2], x[3])
    if isinstance(x, Quaternion):
        return quaternion_to_euler_angles(x.x, x.y, x.z, x.w)

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def pose_to_euler(pose: Pose):
    _, _, yaw = quaternion_to_euler_angles(pose.orientation)
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
