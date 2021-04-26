# rl-navigation
A Reinforcement Learning-based navigation Q-Learning experiment using ROS.

It is recommended that a virtual environment is used while building this project. This ROS package is incompatible with Python2. Please make sure to build the following ROS packages with Python3 before builing this repo with the same Python (Tested with Python 3.7).

- https://github.com/ros-controls/control_toolbox
- https://github.com/ros-simulation/gazebo_ros_pkgs
- https://github.com/ros/geometry
- https://github.com/ros/geometry2
- https://github.com/ros-planning/moveit_msgs
- https://github.com/wg-perception/object_recognition_msgs
- https://github.com/OctoMap/octomap_msgs
- https://github.com/ros-controls/realtime_tools
- https://github.com/ros/ros_comm
- https://github.com/ros-controls/ros_control
- https://github.com/ROBOTIS-GIT/turtlebot3
- https://github.com/ROBOTIS-GIT/turtlebot3_msgs
- https://github.com/ROBOTIS-GIT/turtlebot3_simulation

The python exe can be specified to catkin_make using 
`catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/path/to/python3/bin/python`

Manually install gym-envs using
`pip install -e gym-envs`.

Requires ROS Kinetic and Gazebo. 
