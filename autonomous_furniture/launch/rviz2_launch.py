import os
import sys

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("autonomous_furniture"), "simple_chair.urdf.xacro"]),
            " ",
            "prefix:=chair_ ",
            "connected_to:='' ",
            "xyz:='1 1 0' ",
            "rpy:='0 0 0' ",
            "height:=1.0 ",
            "x_size:=0.5 ",
            "y_size:=0.5"
        ]
    )

    robot_description = {"robot_description": robot_description_content}

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("autonomous_furniture"), "object.rviz"])],
        output="log",
    )

    nodes = [
        robot_state_pub_node,
        rviz_node,
    ]

    return LaunchDescription(nodes)


"""
    LaunchDescription([
        Node(
            package='robot_state_publisher', name='robot_state_publisher', executable='robot_state_publisher',
            parameters=[{'robot_description': urdf}],
        ),
        Node(
            package='rviz2', name='rviz2', executable='rviz2', arguments=['-d', str('~/.rviz2/default.rviz')],

        )
    ])"""

"""
        Node(
            package='robot_state_publisher', name='robot_state_publisher', executable='robot_state_publisher',
            output='screen', arguments=[urdf]
        ),
        Node(
            package='rviz2', name='rviz2', executable='rviz2', arguments=['-d', str('~/.rviz2/default.rviz')],

        )
        
        joint_state_publisher
"""
# parameters=[{'robot_description': urdf}]
