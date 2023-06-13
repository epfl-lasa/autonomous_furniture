""" Launch furniture in Assistive Environment"""
# Author: Lukas Huber
# Created: 2023-01-30
# License: BSD

import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_path

from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    PathJoinSubstitution,
)

from autonomous_furniture.launch_helper_functions import node_creator

def generate_launch_description(n_tables: int = 8):


    nodes = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true",
        ),
    ]
    # return LaunchDescription(nodes + furnite_nodes + wall_nodes + qolo_nodes)
    return LaunchDescription(nodes)