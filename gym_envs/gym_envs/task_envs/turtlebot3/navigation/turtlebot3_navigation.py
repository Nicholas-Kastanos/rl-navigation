import rospy
import numpy as np
from gym import spaces
from ....robot_envs import TurtleBot3Env
from geometry_msgs.msg import Vector3
from ...task_commons import LoadYamlFileParams, pose_to_euler, get_distance, get_diff
from openai_ros.openai_ros_common import ROSLauncher
import os
from geometry_msgs.msg import Pose, Twist
from tf.transformations import quaternion_from_euler

X = 0
Y = 1
YAW = 2


class TurtleBot3NavigationEnv(TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # Load Params from the desired Yaml file
        yaml_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', 'turtlebot3_navigation.yaml')
        LoadYamlFileParams(yaml_file)


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3NavigationEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        self._num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)

        high = np.full((self._num_laser_readings), self.max_laser_value)
        low = np.full((self._num_laser_readings), self.min_laser_value)

        # add goal-odom observation
        ##      x
        high = np.append(high, 8)
        low = np.append(low, -8)

        ##      y
        high = np.append(high, 8)
        low = np.append(low, -8)

        # # Robot current Yaw yaw
        # high = np.append(high, 2 * np.pi)
        # low = np.append(low, 0)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)
        rospy.logwarn("Num Laster Readings: " + str(self._num_laser_readings))
        rospy.logwarn("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logwarn("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.step_reward = rospy.get_param("/turtlebot3/rewards/step")
        self.contact_reward = rospy.get_param("turtlebot3/rewards/contact")

        # Goal Info
        self.goal_distance_tolerance = 0.5
        self.goal = [0, 0, 0]

        self.cumulated_steps = 0.0

    def set_goal(self, x, y):
        self.goal = [x, y]

    def set_tb_state(self, x, y, yaw):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w  = quaternion_from_euler(0, 0, yaw)
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
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """


        rospy.logdebug("Start Set Action ==>"+str(action))
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

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_current_odom_pose(self):
        odom = self.get_odom()
        return pose_to_euler(odom.pose.pose)

    def get_relative_position(self, absolute_pose, absolute_position):
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

    def get_observation(self):
        return self._get_obs()

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )
        current_odom = self._get_current_odom_pose()
        discretized_observations += (self.get_relative_position(current_odom, self.goal) / get_distance(current_odom[2:], self.goal)).tolist()
        # discretized_observations += current_odom[2] # Yaw

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")


        return self.quantise_obs(discretized_observations)

    def _is_at_goal(self):
        distance = get_distance(self._get_current_odom_pose()[:2], self.goal)
        if distance < self.goal_distance_tolerance:
            return True
        return False

    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("TurtleBot3 is already at the destination==>")
        # else:
            # rospy.logwarn("TurtleBot3 is not at the destination==>")

        if self._is_at_goal():
            rospy.logerr("TurtleBot3 reached the destination==>")
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
            reward = self.step_reward # * get_distance(self._get_current_odom_pose()[:2], self.goal)

            # Check if any laser readings are the minimum laser value  ie touching a wall
            mask = np.array(observations[:self._num_laser_readings]) == 0
            if np.any(mask):
                reward += self.contact_reward
        else:
            reward = 0


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

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

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

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