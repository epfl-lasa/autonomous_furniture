import os
import math

from ament_index_python.packages import get_package_share_path

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch.substitutions import (
    Command,
    LaunchConfiguration,
)

import numpy as np

URDF_DIRECTORY = "objects_descriptions"


def node_creator(furniture_name, urdf_file_name, topicspace=""):
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    path_to_urdf = get_package_share_path(URDF_DIRECTORY) / "urdf" / urdf_file_name

    if len(topicspace):
        namespace = topicspace + "/" + furniture_name + "/"
    else:
        namespace = furniture_name

    new_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        namespace=namespace,
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": ParameterValue(
                    Command(
                        [
                            "xacro ",
                            str(path_to_urdf),
                            " ",
                            f"prefix:={furniture_name}/ ",
                            "fixed:='0' ",
                        ]
                    ),
                    value_type=str,
                ),
            }
        ],
        arguments=[str(path_to_urdf)],
    )

    return new_node


def create_wall_node(
    position,
    length: float = 1.0,
    orientation_in_degree: float = 0.0,
    urdf_file_name="wall.urdf.xacro",
    # The 'wall_counter' is shared between all the instances and should not be set
    wall_counter=[0],
):
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    path_to_urdf = get_package_share_path(URDF_DIRECTORY) / "urdf" / urdf_file_name

    rot = orientation_in_degree * math.pi / 180.0
    new_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        namespace="wall" + str(wall_counter[0]),
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": ParameterValue(
                    Command(
                        [
                            "xacro ",
                            str(path_to_urdf),
                            " ",
                            f"prefix:=wall_{str(wall_counter[0])}/ ",
                            "fixed:='1' ",
                            f"xyz:='{position[0]} {position[1]} 1' ",
                            f"dim:='{length + 0.1} 0.1 2' ",
                            f"rpy:='0 0 {rot}' ",
                        ]
                    ),
                    value_type=str,
                ),
            }
        ],
        arguments=[str(path_to_urdf)],
    )

    # Increment counter
    wall_counter[0] += 1

    return new_node


def create_room_with_four_walls(room_axes, center=[0, 0]):
    """Creates a wall with center at center."""

    center_wall_0 = np.array([0, room_axes[1] / 2.0]) + np.array(center)
    center_wall_1 = np.array([room_axes[0] / 2.0, 0]) + np.array(center)
    center_wall_2 = np.array([0, -1 * room_axes[1] / 2.0]) + np.array(center)
    center_wall_3 = np.array([-1 * room_axes[0] / 2.0, 0]) + np.array(center)

    wall_nodes = []
    wall_nodes.append(
        create_wall_node(
            position=center_wall_0,
            length=room_axes[0],
            orientation_in_degree=0.0,
        )
    )
    wall_nodes.append(
        create_wall_node(
            position=center_wall_1,
            length=room_axes[1],
            orientation_in_degree=90.0,
        )
    )
    wall_nodes.append(
        create_wall_node(
            position=center_wall_2,
            length=room_axes[0],
            orientation_in_degree=180.0,
        )
    )
    wall_nodes.append(
        create_wall_node(
            position=center_wall_3,
            length=room_axes[1],
            orientation_in_degree=270.0,
        )
    )
    return wall_nodes


def create_qolo_nodes():
    qolo_nodes = []
    qolo_nodes.append(
        node_creator(furniture_name="qolo", urdf_file_name="wheelchair.urdf.xacro")
    )

    qolo_nodes.append(
        node_creator(
            furniture_name="qolo_human", urdf_file_name="qolo_human.urdf.xacro"
        )
    )
    return qolo_nodes
