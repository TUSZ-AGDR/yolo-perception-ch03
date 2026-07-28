"""
Launch: Gazebo 仿真 + Bridge + YOLO 检测 一条龙

用法:
  ros2 launch yolo_vision sim_detect.launch.py
  ros2 launch yolo_vision sim_detect.launch.py world:=grasping_world.sdf
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('yolo_vision')

    world_arg = DeclareLaunchArgument(
        'world', default_value='block_grid_world.sdf',
        description='World file in yolo_vision/worlds/'
    )
    model_arg = DeclareLaunchArgument(
        'model', default_value='best.pt',
    )
    conf_arg = DeclareLaunchArgument(
        'confidence', default_value='0.5',
    )

    world_path = PathJoinSubstitution([pkg_dir, 'worlds', LaunchConfiguration('world')])

    # ---- Gazebo ----
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen',
    )

    # ---- Bridge (GZ → ROS2) ----
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge_camera',
        arguments=[
            '/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        output='screen',
    )

    # ---- YOLO 检测 ----
    detector = Node(
        package='yolo_vision',
        executable='detector',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'model': LaunchConfiguration('model'),
            'confidence': LaunchConfiguration('confidence'),
        }],
    )

    return LaunchDescription([
        world_arg,
        model_arg,
        conf_arg,
        gz_sim,
        TimerAction(period=3.0, actions=[bridge]),
        TimerAction(period=6.0, actions=[detector]),
    ])
