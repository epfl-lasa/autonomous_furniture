import launch

from autonomous_furniture.launch_generator import generate_launch_description

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


def create_bed_node(name, topicspace="") -> Node:
    return node_creator(
        furniture_name=name,
        urdf_file_name="hospital_bed.urdf.xacro",
        topicspace=topicspace,
    )


def create_table_node(name, topicspace="") -> Node:
    return node_creator(
        furniture_name=name,
        urdf_file_name="table.urdf.xacro",
        topicspace=topicspace,
    )


def create_chair_node(name, topicspace="") -> Node:
    return node_creator(
        furniture_name=name,
        urdf_file_name="chair.urdf.xacro",
        topicspace=topicspace,
    )


def generate_launch_description(
    node_list: list[Node],
    room_axes: list[float] = [14, 11],
    room_center: list[float] = [6, 4.5],
    n_qolos: int = 1,
    create_rviz: bool = False,
):
    wall_nodes = create_room_with_four_walls(room_axes=room_axes, center=room_center)

    qolo_nodes = []
    for ii in range(n_qolos):
        qolo_nodes += create_qolo_nodes()

    # Rviz path -> this could be obtained if correctly installed..
    package_directory = get_package_share_path("autonomous_furniture")

    if create_rviz:
        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=[
                "-d",
                str(package_directory / "config" / "multi_furniture.rviz"),
            ],
            output="log",
        )
        node_list.append(rviz_node)

    nodes = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true",
        ),
    ]
    # return LaunchDescription(nodes + furnite_nodes + wall_nodes + qolo_nodes)
    return LaunchDescription(nodes + node_list + wall_nodes + qolo_nodes)


def main():
    nodes = []
    for ii in range(6):
        nodes.append(create_table_node(name=f"furniture{ii}"))

    launch_description = generate_launch_description(nodes, create_rviz=True)
    launch_service = launch.LaunchService()
    launch_service.include_launch_description(launch_description)
    launch_service.run()


if (__name__) == "__main__":
    main()
    print("Done")
