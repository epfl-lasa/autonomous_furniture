from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    qolo_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("furniture_descriptions"), "urdf/qolo_human.urdf.xacro"]
            ),
            " ",
            "prefix:=qolo_human_ ",
            "fixed:='0' ",
        ]
    )
    qolo_description = {
        "use_sim_time": use_sim_time,
        "robot_description": qolo_description_content,
    }

    h_bed_1_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("furniture_descriptions"),
                    "urdf/hospital_bed.urdf.xacro",
                ]
            ),
            " ",
            "prefix:=h_bed_1_ ",
            "fixed:='0' ",
            "rpy:='-1.57 0 0' ",
        ]
    )
    h_bed_1_description = {
        "use_sim_time": use_sim_time,
        "robot_description": h_bed_1_description_content,
    }

    h_bed_2_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("furniture_descriptions"),
                    "urdf/hospital_bed.urdf.xacro",
                ]
            ),
            " ",
            "prefix:=h_bed_2_ ",
            "fixed:='0' ",
            "rpy:='-1.57 0 0' ",
        ]
    )
    h_bed_2_description = {
        "use_sim_time": use_sim_time,
        "robot_description": h_bed_2_description_content,
    }

    h_bed_3_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("furniture_descriptions"),
                    "urdf/hospital_bed.urdf.xacro",
                ]
            ),
            " ",
            "prefix:=h_bed_3_ ",
            "fixed:='0' ",
            "rpy:='-1.57 0 0' ",
        ]
    )
    h_bed_3_description = {
        "use_sim_time": use_sim_time,
        "robot_description": h_bed_3_description_content,
    }

    h_bed_4_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("furniture_descriptions"),
                    "urdf/hospital_bed.urdf.xacro",
                ]
            ),
            " ",
            "prefix:=h_bed_4_ ",
            "fixed:='0' ",
            "rpy:='-1.57 0 0' ",
        ]
    )
    h_bed_4_description = {
        "use_sim_time": use_sim_time,
        "robot_description": h_bed_4_description_content,
    }

    h_bed_5_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("furniture_descriptions"),
                    "urdf/hospital_bed.urdf.xacro",
                ]
            ),
            " ",
            "prefix:=h_bed_5_ ",
            "fixed:='0' ",
            "rpy:='-1.57 0 0' ",
        ]
    )
    h_bed_5_description = {
        "use_sim_time": use_sim_time,
        "robot_description": h_bed_5_description_content,
    }

    h_bed_6_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("furniture_descriptions"),
                    "urdf/hospital_bed.urdf.xacro",
                ]
            ),
            " ",
            "prefix:=h_bed_6_ ",
            "fixed:='0' ",
            "rpy:='-1.57 0 0' ",
        ]
    )
    h_bed_6_description = {
        "use_sim_time": use_sim_time,
        "robot_description": h_bed_6_description_content,
    }

    wall_1_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("furniture_descriptions"), "urdf/wall.urdf.xacro"]
            ),
            " ",
            "prefix:=wall_1_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='0 5 1' ",
            "rpy:='0 0 0' ",
            "dim:='12 0.1 2' ",
        ]
    )
    wall_1_description = {
        "use_sim_time": use_sim_time,
        "robot_description": wall_1_description_content,
    }

    wall_2_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("furniture_descriptions"), "urdf/wall.urdf.xacro"]
            ),
            " ",
            "prefix:=wall_2_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='6 0 0.5' ",
            "rpy:='0 0 0' ",
            "dim:='0.1 10 1' ",
        ]
    )
    wall_2_description = {
        "use_sim_time": use_sim_time,
        "robot_description": wall_2_description_content,
    }

    wall_3_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("furniture_descriptions"), "urdf/wall.urdf.xacro"]
            ),
            " ",
            "prefix:=wall_3_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='0 -5 0.5' ",
            "rpy:='0 0 0' ",
            "dim:='12 0.1 1' ",
        ]
    )
    wall_3_description = {
        "use_sim_time": use_sim_time,
        "robot_description": wall_3_description_content,
    }

    wall_4_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("furniture_descriptions"), "urdf/wall.urdf.xacro"]
            ),
            " ",
            "prefix:=wall_4_ ",
            "fixed:='1' ",
            "connected_to:='odom' ",
            "xyz:='-6 0 1' ",
            "rpy:='0 0 0' ",
            "dim:='0.1 10 2' ",
        ]
    )
    wall_4_description = {
        "use_sim_time": use_sim_time,
        "robot_description": wall_4_description_content,
    }

    qolo_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="qolo_human",
        executable="robot_state_publisher",
        output="both",
        parameters=[qolo_description],
    )

    h_bed_1_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="h_bed_1_",
        executable="robot_state_publisher",
        output="both",
        parameters=[h_bed_1_description],
    )

    h_bed_2_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="h_bed_2_",
        executable="robot_state_publisher",
        output="both",
        parameters=[h_bed_2_description],
    )

    h_bed_3_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="h_bed_3_",
        executable="robot_state_publisher",
        output="both",
        parameters=[h_bed_3_description],
    )

    h_bed_4_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="h_bed_4_",
        executable="robot_state_publisher",
        output="both",
        parameters=[h_bed_4_description],
    )

    h_bed_5_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="h_bed_5_",
        executable="robot_state_publisher",
        output="both",
        parameters=[h_bed_5_description],
    )

    h_bed_6_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="h_bed_6_",
        executable="robot_state_publisher",
        output="both",
        parameters=[h_bed_6_description],
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
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("furniture_descriptions"), "rviz/hos_env.rviz"]
            ),
        ],
        output="log",
    )

    sim_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    nodes = [
        sim_arg,
        qolo_state_pub_node,
        h_bed_1_state_pub_node,
        h_bed_2_state_pub_node,
        h_bed_3_state_pub_node,
        h_bed_4_state_pub_node,
        # h_bed_5_state_pub_node,
        # h_bed_6_state_pub_node,
        wall_1_state_pub_node,
        wall_2_state_pub_node,
        wall_3_state_pub_node,
        wall_4_state_pub_node,
        rviz_node,
    ]

    return LaunchDescription(nodes)
