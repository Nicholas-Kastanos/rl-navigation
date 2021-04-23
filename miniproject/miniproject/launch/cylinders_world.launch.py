import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def get_urdf(model_name: str):
    urdf_file_name = 'turtlebot3_' + model_name + '.urdf'
    print('urdf_file_name : {}'.format(urdf_file_name))

    parameters = [
        {
            "turtlebot3" : {
                "linear_forward_speed": 0.5,
                "linear_turn_speed": 0.5,
                "angular_speed": 0.25,
                "init_linear_forward_speed": 0.0,
                "init_linear_turn_speed": 0.0,
                "new_ranges": 25,
                "min_range": 0.2,
                "max_laser_value": 3.5,
                "min_laser_value": 0.0
            }
        }
    ]
    urdf = os.path.join(
        get_package_share_directory('turtlebot3_description'),
        'urdf',
        urdf_file_name)
    return urdf, parameters

def get_qlearn_params():
    return [
        {
            "environment": {
                "task_and_robot_environment_name": 'TurtleBot3Navigation-v0',
                "ros_ws_abspath": "/home/nicholas/project_cws"
            },
            "qlearning": {
                "n_actions": 6,
                "alpha": 0.5,
                "gamma": 0.7,
                "epsilon": 0.995,
                "epsilon_discount": 0.995,
                "nepisodes": 2000,
                "nsteps": 200,
                "rewards": {
                    "step": -1,
                    "contact": -10
                }
            }
        }
    ]

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    world_file_name = 'cylinders_world.model'
    world = os.path.join(get_package_share_directory('miniproject'),
                         'worlds', world_file_name)
    
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    parameters = [{'use_sim_time': use_sim_time}]

    urdf, tb_params = get_urdf('burger')
    qlearn_params = get_qlearn_params()
    parameters += tb_params + qlearn_params

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
        ),
        Node(
            package='miniproject',
            executable='start_qlearning',
            name='start_qlearning',
            output='screen',
            parameters=parameters,
            arguments=[urdf]),
    ])
