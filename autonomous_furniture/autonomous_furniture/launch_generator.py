""" Launch furniture in Assistive Environment"""
# Author: Lukas Huber
# Created: 2023-01-30
# License: BSD

from dataclasses import dataclass
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_path
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_path

from autonomous_furniture.launch_helper_functions import create_room_with_four_walls
from autonomous_furniture.launch_helper_functions import create_qolo_nodes
from autonomous_furniture.launch_helper_functions import node_creator


@dataclass
class FurnitureNodeInfo:
    name: str
    urdf_file: str
    topicspace: str = "furniture"


def create_bed_node(name) -> Node:
    return node_creator(
        furniture_name=name,
        urdf_file_name="hospital_bed.urdf.xacro",
        topicspace="furniture",
    )


def create_table_node(name) -> Node:
    return node_creator(
        furniture_name=name,
        urdf_file_name="table.urdf.xacro",
        topicspace="furniture",
    )


def create_chair_node(name) -> Node:
    return node_creator(
        furniture_name=name,
        urdf_file_name="chair.urdf.xacro",
        topicspace="furniture",
    )


def generate_launch_description(
    node_list: list[Node],
    room_axes: list[float] = [14, 11],
    room_center: list[float] = [6, 4.5],
    n_qolos: int = 1,
):
    wall_nodes = create_room_with_four_walls(room_axes=room_axes, center=room_center)

    qolo_nodes = []
    for ii in range(n_qolos):
        qolo_nodes += create_qolo_nodes()

    # Rviz path -> this could be obtained if correctly installed..
    package_directory = get_package_share_path("autonomous_furniture")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", str(package_directory / "config" / "multi_bed.rviz")],
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
    return LaunchDescription(nodes + node_list + wall_nodes + qolo_nodes)
