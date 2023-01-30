import os
import math

from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import get_package_share_path

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch.substitutions import (
    Command,
    LaunchConfiguration,
)


def node_creator(furniture_name, urdf_file_name):
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    path_to_urdf = (
        get_package_share_path("objects_descriptions") / "urdf" / urdf_file_name
    )

    urdf = os.path.join(
        get_package_share_directory("objects_descriptions"),
        os.path.join("urdf", urdf_file_name),
    )

    new_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        namespace="furniture" + "/" + furniture_name,
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": ParameterValue(
                    Command(
                        [
                            "xacro ",
                            str(path_to_urdf),
                            " ",
                            f"prefix:={furniture_name} ",
                            "fixed:='0' ",
                        ]
                    ),
                    value_type=str,
                ),
            }
        ],
        arguments=[urdf],
    )

    return new_node


def create_wall_node(
    position,
    length: float = 1.0,
    orientation_in_degree: float = 0.0,
    urdf_file_name="wall.urdf.xacro",
    # This is shared between all the instances and should not be set
    wall_counter=[0],
):
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    path_to_urdf = (
        get_package_share_path("objects_descriptions") / "urdf" / urdf_file_name
    )

    urdf = os.path.join(
        get_package_share_directory("objects_descriptions"),
        os.path.join("urdf", urdf_file_name),
    )

    rot = orientation_in_degree * math.pi / 180.0
    new_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        namespace="furniture" + "/" + "wall" + str(wall_counter[0]),
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": ParameterValue(
                    Command(
                        [
                            "xacro ",
                            str(path_to_urdf),
                            " ",
                            f"prefix:=wall_{str(wall_counter[0])} ",
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
        arguments=[urdf],
    )

    # Increment counter
    wall_counter[0] += 1

    return new_node


def create_room_with_four_walls(room_axes):
    """Creates a wall with center at [0,0]."""

    wall_nodes = []
    wall_nodes.append(
        create_wall_node(
            position=[0, room_axes[1] / 2.0],
            length=room_axes[0],
            orientation_in_degree=00.0,
        )
    )
    wall_nodes.append(
        create_wall_node(
            position=[room_axes[0] / 2.0, 0],
            length=room_axes[1],
            orientation_in_degree=90.0,
        )
    )
    wall_nodes.append(
        create_wall_node(
            position=[0, -1 * room_axes[1] / 2.0],
            length=room_axes[0],
            orientation_in_degree=180.0,
        )
    )
    wall_nodes.append(
        create_wall_node(
            position=[-1 * room_axes[0] / 2.0, 0],
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
