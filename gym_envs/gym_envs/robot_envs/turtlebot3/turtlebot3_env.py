import threading

import numpy
import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from openai_ros2 import RobotGazeboEnv, exceptions, service_utils
from openai_ros2.rate_runner import WhileRateRunner, ForRateRunner
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image, Imu, JointState, LaserScan, PointCloud2
from std_msgs.msg import Float64


class TurtleBot3Env(RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, node: Node, ros_ws_abspath):
        """
        Initializes a new TurtleBot3Env environment.
        TurtleBot3 doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /imu: Inertial Mesuring Unit that gives relative accelerations and orientations.
        * /scan: Laser Readings

        Actuators Topic List: /cmd_vel,

        Args:
        """

        node.get_logger().debug("Start TurtleBot3Env INIT...")

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = ["imu"]

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtleBot3Env, self).__init__(node=node,
                                            controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False)
        # Variables that we give through the constructor.
        # None in this case

        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()

        # We Start all the ROS related Subscribers and publishers
        self.odom_sub = self.node.create_subscription(Odometry, "/odom", self._odom_callback, 1)
        self.imu_sub = self.node.create_subscription(Imu, "/imu", self._imu_callback, 1)
        self.scan_sub = self.node.create_subscription(LaserScan, "/scan", self._laser_scan_callback, 1)
        
        self._check_all_sensors_ready() # Sensors must be checked after the callbacks are created
    
        self._cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 1)

        self._check_publishers_connection()

        self._set_entity_state_client = service_utils.create_service_client(self.node, SetEntityState, '/gazebo_ros_state/set_entity_state')
        
        self.gazebo.pauseSim()

        self._logger.debug("Finished TurtleBot3Env INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self._logger.debug("START ALL SENSORS READY")
        self._check_odom_ready()
        self._check_imu_ready()
        self._check_laser_scan_ready()
        self._logger.debug("FINISH ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        self._logger.debug("Waiting for /odom to be READY...")
        WhileRateRunner(
            self.node, 
            frequency=20,
            cond=lambda: self.odom is None,
            func=lambda: self._logger.debug("Waiting for /odom to be READY... Not Yet")
        )
        self._logger.debug("Waiting for /odom to be READY... READY!")


    def _check_imu_ready(self):
        self.imu = None
        self._logger.debug("Waiting for /imu to be READY...")
        WhileRateRunner(
            self.node, 
            frequency=20,
            cond=lambda: self.imu is None,
            func=lambda: self._logger.debug("Waiting for /imu to be READY... Not Yet")
        )
        self._logger.debug("Waiting for /imu to be READY... READY!")

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        self._logger.debug("Waiting for /scan to be READY...")
        WhileRateRunner(
            self.node, 
            frequency=20,
            cond=lambda: self.laser_scan is None,
            func=lambda: self._logger.debug("Waiting for /scan to be READY... Not Yet")
        )
        self._logger.debug("Waiting for /scan to be READY... READY!")


    def _odom_callback(self, data):
        self.odom = data

    def _imu_callback(self, data):
        self.imu = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        WhileRateRunner(
            self.node,
            frequency=10,
            func=lambda: self._logger.debug("No susbribers to _cmd_vel_pub yet so we wait and try again"),
            cond=lambda: self._cmd_vel_pub.get_subscription_count() == 0
        )
        self._logger.debug("_cmd_vel_pub Publisher Connected")
        self._logger.debug("All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def _send_move_msg(self, linear_speed, angular_speed):
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        self._logger.debug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)

    def move_base(self, linear_speed, angular_speed, move_freq=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :return:
        """
        ForRateRunner(
            self.node,
            frequency=move_freq,
            func=self._send_move_msg(linear_speed, angular_speed),
            num_loops=1
        )

    def _set_entity_state(self, name: str, pose: Pose=None, twist:Twist=None, reference_frame:str="world"):
        set_entity_state_req = SetEntityState.Request()
        set_entity_state_req.state.name=name
        set_entity_state_req.state.pose=pose
        set_entity_state_req.state.twist=twist
        set_entity_state_req.state.reference_frame=reference_frame
        self._logger.debug("TurtleBot3 EntityState Cmd>>" + str(set_entity_state_req))
        service_utils.call_and_wait_for_service_response(self.node, self._set_entity_state_client, set_entity_state_req)

    def get_odom(self):
        return self.odom

    def get_imu(self):
        return self.imu

    def get_laser_scan(self):
        return self.laser_scan
