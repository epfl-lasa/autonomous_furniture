""" Launch furniture in Assistive Environment"""
# Author: Lukas Huber
# Created: 2023-01-30
# License: BSD

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    PathJoinSubstitution,
)

from autonomous_furniture.launch_helper_functions import node_creator
from autonomous_furniture.launch_helper_functions import create_room_with_four_walls
from autonomous_furniture.launch_helper_functions import create_qolo_nodes


def generate_launch_description():
    furnite_nodes = []
    furnite_nodes.append(
        node_creator(furniture_name="table", urdf_file_name="table.urdf.xacro")
    )

    n_chairs = 2
    for ii in range(n_chairs):
        furnite_nodes.append(
            node_creator(
                furniture_name="chair" + str(ii), urdf_file_name="chair.urdf.xacro"
            )
        )

    wall_nodes = create_room_with_four_walls(room_axes=[7, 10])
    qolo_nodes = create_qolo_nodes()

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [
                    FindPackageShare("autonomous_furniture"),
                    "rviz/assistive_environment.rviz",
                ]
            ),
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
    return LaunchDescription(nodes + furnite_nodes + wall_nodes + qolo_nodes)
