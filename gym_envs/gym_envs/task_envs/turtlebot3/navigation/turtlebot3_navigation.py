import os

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist, Vector3
from gym import spaces
from rcl_interfaces.msg import ParameterType, ParameterDescriptor

from ....robot_envs import TurtleBot3Env
from ...task_commons import (euler_angles_to_quaternion,
                             get_distance, get_relative_position,
                             pose_to_euler, signed_yaw_diff)


class TurtleBot3NavigationEnv(TurtleBot3Env):
    def __init__(self, node):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # Load Params from the desired Yaml file
        # yaml_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', 'turtlebot3_navigation.yaml')
        # LoadYamlFileParams(yaml_file)

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = node.declare_parameter("environment.ros_ws_abspath").value
        assert isinstance(ros_ws_abspath, str)
        
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3NavigationEnv, self).__init__(node, ros_ws_abspath)


        # Only variable needed to be set here
        number_actions = self.node.declare_parameter("qlearning.n_actions").value
        assert isinstance(number_actions, int)

        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # Actions and Observations
        self.linear_forward_speed = self.node.declare_parameter('turtlebot3.linear_forward_speed').value
        self.linear_turn_speed = self.node.declare_parameter('turtlebot3.linear_turn_speed').value
        self.angular_speed = self.node.declare_parameter('turtlebot3.angular_speed').value
        self.init_linear_forward_speed = self.node.declare_parameter('turtlebot3.init_linear_forward_speed').value
        self.init_linear_turn_speed = self.node.declare_parameter('turtlebot3.init_linear_turn_speed').value

        self.new_ranges = self.node.declare_parameter('turtlebot3.new_ranges').value
        self.min_range = self.node.declare_parameter('turtlebot3.min_range').value
        self.max_laser_value = self.node.declare_parameter('turtlebot3.max_laser_value').value
        self.min_laser_value = self.node.declare_parameter('turtlebot3.min_laser_value').value

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        self._num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)

        high = np.full((self._num_laser_readings), self.max_laser_value)
        low = np.full((self._num_laser_readings), self.min_laser_value)

        # relative position of goal (only x and y) relative to robot
        ##      x
        high = np.append(high, 8)
        low = np.append(low, -8)

        ##      y
        high = np.append(high, 8)
        low = np.append(low, -8)

        # difference in yaw between robot and desired yaw
        high = np.append(high, np.pi)
        low = np.append(low, -np.pi)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)
        self._logger.warning("Num Laster Readings: " + str(self._num_laser_readings))
        self._logger.warning("ACTION SPACES TYPE===>"+str(self.action_space))
        self._logger.warning("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.step_reward = self.node.declare_parameter("qlearning.rewards.step").value
        self.contact_reward = self.node.declare_parameter("qlearning.rewards.contact").value

        # Goal Info
        self.goal_distance_tolerance = 0.2
        self.set_goal(0, 0, 0)

        self.cumulated_steps = 0.0

    def set_goal(self, x, y, yaw):
        self.goal = [x, y, np.mod(yaw, 2*np.pi)]

    def set_tb_state(self, x, y, yaw):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w  = euler_angles_to_quaternion(0, 0, yaw)
        self._set_model_state("turtlebot3_burger", pose=pose, twist=Twist())


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set done to false, because its calculated asyncronously
        self._episode_done = False

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        self._loggerdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1:
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "FORWARD_RIGHT"
        elif action == 2:
            linear_speed = 0.0
            angular_speed = -1*self.angular_speed
            self.last_action = "RIGHT"
        elif action == 3:
            linear_speed = -1*self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "REVERSE"
        elif action == 4:
            linear_speed = 0.0
            angular_speed = self.angular_speed
            self.last_action = "LEFT"
        elif action == 5:
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "FORWARD_LEFT"
        self.move_base(linear_speed, angular_speed)

        self._logger.debug("END Set Action ==>"+str(action))

    def _get_current_odom_pose(self):
        odom = self.get_odom()
        return pose_to_euler(odom.pose.pose)

    def get_observation(self):
        return self._get_obs()

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        self._logger.debug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )
        current_odom = self._get_current_odom_pose()
        discretized_observations += get_relative_position(current_odom, self.goal).tolist()
        discretized_observations += [signed_yaw_diff(current_odom, self.goal)] # Yaw

        self._logger.debug("Observations==>"+str(discretized_observations))
        self._logger.debug("END Get Observation ==>")

        return self.quantise_obs(discretized_observations)

    def _is_at_goal(self):
        distance = get_distance(self._get_current_odom_pose(), self.goal)
        if distance < self.goal_distance_tolerance:
            return True
        return False

    def _is_done(self, observations):

        if self._episode_done:
            self._logger.error("TurtleBot3 is already at the destination==>")
        # else:
            # rospy.logwarn("TurtleBot3 is not at the destination==>")

        if self._is_at_goal():
            self._logger.error("TurtleBot3 reached the destination==>")
            self._episode_done = True
        

        # # Now we check if it has crashed based on the imu
        # imu_data = self.get_imu()
        # linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        # if linear_acceleration_magnitude > self.max_linear_aceleration:
        #     rospy.logerr("TurtleBot2 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
        #     self._episode_done = True
        # else:
        #     rospy.logerr("DIDNT crash TurtleBot2 ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))


        return self._episode_done

    def _compute_reward(self, observations, done):

        if not done:
            # if self.last_action == "FORWARDS":
            #     reward = self.forwards_reward
            # else:
            #     reward = self.turn_reward
            reward = self.step_reward # * get_distance(self._get_current_odom_pose(), self.goal)

            # Check if any laser readings are the minimum laser value  ie touching a wall
            mask = np.array(observations[:self._num_laser_readings]) == 0
            if np.any(mask):
                reward += self.contact_reward
        else:
            reward = 0


        self._logger.debug("reward=" + str(reward))
        self.cumulated_reward += reward
        self._logger.debug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        self._logger.debug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods
    def discretize_scan_observation(self,data,new_ranges): 
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        self._logger.debug("data=" + str(data))
        self._logger.debug("new_ranges=" + str(new_ranges))
        self._logger.debug("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                # if (self.min_range > item > 0):
                #     rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                #     self._episode_done = True
                # else:
                #     rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges

    def quantise_obs(self, obs):
        return np.around(obs, decimals=1)

    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = np.array((vector.x, vector.y, vector.z))
        force_magnitude = np.linalg.norm(contact_force_np)

        return force_magnitude
