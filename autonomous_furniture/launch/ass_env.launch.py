from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    table_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/table.urdf.xacro"]),
            " ",
            "prefix:=table_ ",
            "fixed:='0' ",
        ]
    )
    table_description = {"use_sim_time": use_sim_time, "robot_description": table_description_content}

    chair1_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/chair.urdf.xacro"]),
            " ",
            "prefix:=chair_1_ ",
            "fixed:='0' ",
        ]
    )
    chair1_description = {"use_sim_time": use_sim_time, "robot_description": chair1_description_content}

    chair2_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/chair.urdf.xacro"]),
            " ",
            "prefix:=chair_2_ ",
            "fixed:='0' ",
        ]
    )
    chair2_description = {"use_sim_time": use_sim_time, "robot_description": chair2_description_content}

    chair3_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/chair.urdf.xacro"]),
            " ",
            "prefix:=chair_3_ ",
            "fixed:='0' ",
        ]
    )
    chair3_description = {"use_sim_time": use_sim_time, "robot_description": chair3_description_content}

    chair4_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/chair.urdf.xacro"]),
            " ",
            "prefix:=chair_4_ ",
            "fixed:='0' ",
        ]
    )
    chair4_description = {"use_sim_time": use_sim_time, "robot_description": chair4_description_content}

    wheelchair_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/wheelchair.urdf.xacro"]),
            " ",
            "prefix:=wheelchair_ ",
            "fixed:='0' ",
        ]
    )
    wheelchair_description = {"use_sim_time": use_sim_time, "robot_description": wheelchair_description_content}

    qolo_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/qolo.urdf.xacro"]),
            " ",
            "prefix:=qolo_ ",
            "fixed:='0' ",
        ]
    )
    qolo_description = {"use_sim_time": use_sim_time, "robot_description": qolo_description_content}

    wall_1_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/wall.urdf.xacro"]),
            " ",
            "prefix:=wall_1_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='0 4 1' ",
            "rpy:='0 0 0' ",
            "dim:='10 0.1 2' ",
        ]
    )
    wall_1_description = {"use_sim_time": use_sim_time, "robot_description": wall_1_description_content}

    wall_2_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/wall.urdf.xacro"]),
            " ",
            "prefix:=wall_2_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='5 0 0.5' ",
            "rpy:='0 0 0' ",
            "dim:='0.1 8 1' ",
        ]
    )
    wall_2_description = {"use_sim_time": use_sim_time, "robot_description": wall_2_description_content}

    wall_3_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/wall.urdf.xacro"]),
            " ",
            "prefix:=wall_3_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='0 -4 0.5' ",
            "rpy:='0 0 0' ",
            "dim:='10 0.1 1' ",
        ]
    )
    wall_3_description = {"use_sim_time": use_sim_time, "robot_description": wall_3_description_content}

    wall_4_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/wall.urdf.xacro"]),
            " ",
            "prefix:=wall_4_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='-5 0 1' ",
            "rpy:='0 0 0' ",
            "dim:='0.1 8 2' ",
        ]
    )
    wall_4_description = {"use_sim_time": use_sim_time, "robot_description": wall_4_description_content}

    table_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="table_",
        executable="robot_state_publisher",
        output="both",
        parameters=[table_description],
    )

    chair1_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="chair_1_",
        executable="robot_state_publisher",
        output="both",
        parameters=[chair1_description],
    )

    chair2_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="chair_2_",
        executable="robot_state_publisher",
        output="both",
        parameters=[chair2_description],
    )

    chair3_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="chair_3_",
        executable="robot_state_publisher",
        output="both",
        parameters=[chair3_description],
    )

    chair4_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="chair_4_",
        executable="robot_state_publisher",
        output="both",
        parameters=[chair4_description],
    )

    qolo_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="qolo",
        executable="robot_state_publisher",
        output="both",
        parameters=[qolo_description],
    )

    wall_1_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="wall_1_",
        executable="robot_state_publisher",
        output="both",
        parameters=[wall_1_description],
    )

    wall_2_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="wall_2_",
        executable="robot_state_publisher",
        output="both",
        parameters=[wall_2_description],
    )

    wall_3_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="wall_3_",
        executable="robot_state_publisher",
        output="both",
        parameters=[wall_3_description],
    )

    wall_4_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="wall_4_",
        executable="robot_state_publisher",
        output="both",
        parameters=[wall_4_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("objects_descriptions"), "rviz/ass_env.rviz"])],
        output="log",
    )

    sim_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    nodes = [
        sim_arg,
        table_state_pub_node,
        chair1_state_pub_node,
        chair2_state_pub_node,
        chair3_state_pub_node,
        chair4_state_pub_node,
        qolo_state_pub_node,
        wall_1_state_pub_node,
        wall_2_state_pub_node,
        wall_3_state_pub_node,
        wall_4_state_pub_node,
        rviz_node,
    ]

    return LaunchDescription(nodes)
