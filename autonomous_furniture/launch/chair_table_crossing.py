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
    furnite_nodes = []
    furnite_nodes.append(
        node_creator(
            furniture_name="chair_down",
            urdf_file_name="chair.urdf.xacro",
            topicspace="furniture",
        )
    )
    furnite_nodes.append(
        node_creator(
            furniture_name="chair_up",
            urdf_file_name="chair.urdf.xacro",
            topicspace="furniture",
        )
    )
    furnite_nodes.append(
        node_creator(
            furniture_name="table",
            urdf_file_name="table.urdf.xacro",
            topicspace="furniture",
        )
    )

    # Rviz path -> this could be obtained if correctly installed..
    rviz_base_path = "/home/" + os.getlogin() + "/ros2_ws/src/autonomous_furniture"

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            os.path.join(rviz_base_path, "config", "chair_table_crossing.rviz")
            # PathJoinSubstitution(
            #     [
            #         FindPackageShare("autonomous_furniture"),
            #         "config/assistive_environment.rviz",
            #     ]
            # ),
        ],
        output="log",
    )

    nodes = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true",
        ),
        rviz_node,
    ]
    # return LaunchDescription(nodes + furnite_nodes + wall_nodes + qolo_nodes)
    return LaunchDescription(nodes + furnite_nodes)