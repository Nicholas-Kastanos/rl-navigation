import numpy
import rclpy
from openai_ros2 import RobotGazeboEnv, exceptions, service_utils
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState

class TurtleBot3Env(RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, node, ros_ws_abspath):
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


        # rospy.logdebug("Start TurtleBot3Env INIT...")
        self._logger.debug("Start TurtleBot3Env INIT...")
        # Variables that we give through the constructor.
        # None in this case


        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        self.odo_sub = self.node.create_subscription(Odometry, "/odom", self._odom_callback, 1)
        self.imu_sub = self.node.create_subscription(Imu, "/imu", self._imu_callback, 1)
        self.scan_sub = self.node.create_subscription(LaserScan, "/scan", self._laser_scan_callback, 1)

        self._cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 1)
        self._set_model_state_pub = self.node.create_publisher(ModelState, 'gazebo/set_model_state', 1)

        self._check_publishers_connection()

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
        self._logger.debug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        self._logger.debug("Waiting for /odom to be READY...")

        _rate = self.node.create_rate(20)

        while self.odom is None:
            try:
                _rate.sleep()
            except rclpy.exceptions.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        self._logger.debug("Waiting for /odom to be READY... READY!")
        _rate.destroy()


    def _check_imu_ready(self):
        self.imu = None
        self._logger.debug("Waiting for /imu to be READY...")
        _rate = self.node.create_rate(20)
        while self.imu is None:
            try:
                _rate.sleep()
            except rclpy.exceptions.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        self._logger.debug("Waiting for /imu to be READY... READY!")
        _rate.destroy()

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        self._logger.debug("Waiting for /scan to be READY...")
        _rate = self.node.create_rate(20)
        while self.laser_scan is None:
            try:
                _rate.sleep()
            except rclpy.exceptions.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        self._logger.debug("Waiting for /scan to be READY... READY!")
        _rate.destroy()


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
        rate = self.node.create_rate(10)  # 10hz
        while self._cmd_vel_pub.get_subscription_count() == 0:
            self._logger.debug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rclpy.exceptions.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        self._logger.debug("_cmd_vel_pub Publisher Connected")

        while self._set_model_state_pub.get_subscription_count() == 0:
            self._logger.debug("No susbribers to _set_model_state_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rclpy.exceptions.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        self._logger.debug("_set_model_state_pub Publisher Connected")

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
    def move_base(self, linear_speed, angular_speed, move_freq=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :return:
        """
        rate = self.node.create_rate(move_freq)
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        self._logger.debug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        try:
            rate.sleep()
        except rclpy.exceptions.ROSInterruptException:
            # This is to avoid error when world is rested, time when backwards.
            pass

    def _set_model_state(self, model_name: str, pose: Pose=None, twist:Twist=None, reference_frame:str="world"):
        model_state_msg = ModelState()
        model_state_msg.model_name = model_name
        model_state_msg.pose = pose
        model_state_msg.twist = twist
        model_state_msg.reference_frame = reference_frame
        self._logger.debug("TurtleBot3 ModelState Cmd>>" + str(model_state_msg))
        self._check_publishers_connection()
        self._set_model_state_pub.publish(model_state_msg)

    def get_odom(self):
        return self.odom

    def get_imu(self):
        return self.imu

    def get_laser_scan(self):
        return self.laser_scan