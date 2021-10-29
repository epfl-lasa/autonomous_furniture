from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/chair.urdf.xacro"]),
            " ",
            "prefix:=chair_ ",
            "connected_to:='' ",
            "xyz:='0 0 0' ",
            "rpy:='0 0 0' ",
        ]
    )
    robot_description = {"use_sim_time": use_sim_time, "robot_description": robot_description_content}

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    state_pub_node = Node(
        package='autonomous_furniture',
        executable='state_publisher',
        name='state_publisher',
        output='screen',
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("objects_descriptions"), "rviz/object_move.rviz"])],
        output="log",
    )

    sim_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    nodes = [
        sim_arg,
        robot_state_pub_node,
        state_pub_node,
        rviz_node,
    ]

    return LaunchDescription(nodes)
