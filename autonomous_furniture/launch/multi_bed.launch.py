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
from autonomous_furniture.launch_helper_functions import create_room_with_four_walls
from autonomous_furniture.launch_helper_functions import create_qolo_nodes


def generate_launch_description():
    furnite_nodes = []

    n_tables = 8
    for ii in range(n_tables):
        furnite_nodes.append(
            node_creator(
                furniture_name=f"hospital_bed{ii}",
                urdf_file_name="hospital_bed.urdf.xacro",
                topicspace="furniture",
            )
        )

    wall_nodes = create_room_with_four_walls(room_axes=[14, 11], center=[6, 4.5])
    qolo_nodes = create_qolo_nodes()

    # Rviz path -> this could be obtained if correctly installed..
    rviz_base_path = "/home/" + os.getlogin() + "/ros2_ws/src/autonomous_furniture"

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            os.path.join(rviz_base_path, "config", "multi_bed.rviz")
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
    return LaunchDescription(nodes + furnite_nodes + wall_nodes + qolo_nodes)
