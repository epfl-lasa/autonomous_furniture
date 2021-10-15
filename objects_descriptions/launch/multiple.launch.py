from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    table_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/table.urdf.xacro"]),
            " ",
            "prefix:=table_ ",
            "connected_to:='' ",
            "xyz:='0 0 0' ",
            "rpy:='0 0 0' ",
        ]
    )
    table_description = {"robot_description": table_description_content}

    chair1_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/chair.urdf.xacro"]),
            " ",
            "prefix:=chair1_ ",
            "connected_to:='' ",
            "xyz:='0 -1 0' ",
            "rpy:='0 0 1.570796327' ",
        ]
    )
    chair1_description = {"robot_description": chair1_description_content}

    chair2_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/chair.urdf.xacro"]),
            " ",
            "prefix:=chair2_ ",
            "connected_to:='' ",
            "xyz:='0 1 0' ",
            "rpy:='0 0 -1.570796327' ",
        ]
    )
    chair2_description = {"robot_description": chair2_description_content}

    wheelchair_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/wheelchair.urdf.xacro"]),
            " ",
            "prefix:=wheelchair_ ",
            "connected_to:='' ",
            "xyz:='-1.5 0 0' ",
            "rpy:='0 0 0' ",
        ]
    )
    wheelchair_description = {"robot_description": wheelchair_description_content}

    qolo_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/qolo.urdf.xacro"]),
            " ",
            "prefix:=qolo_ ",
            "connected_to:='' ",
            "xyz:='-1.5 0 0' ",
            "rpy:='0 0 0' ",
        ]
    )
    qolo_description = {"robot_description": qolo_description_content}

    table_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="table",
        executable="robot_state_publisher",
        output="both",
        parameters=[table_description],
    )

    chair1_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="chair1",
        executable="robot_state_publisher",
        output="both",
        parameters=[chair1_description],
    )

    chair2_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="chair2",
        executable="robot_state_publisher",
        output="both",
        parameters=[chair2_description],
    )

    wheelchair_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="wheelchair",
        executable="robot_state_publisher",
        output="both",
        parameters=[wheelchair_description],
    )

    qolo_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="qolo",
        executable="robot_state_publisher",
        output="both",
        parameters=[qolo_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("objects_descriptions"), "rviz/multiple_obj.rviz"])],
        output="log",
    )

    nodes = [
        table_state_pub_node,
        chair1_state_pub_node,
        chair2_state_pub_node,
        # wheelchair_state_pub_node,
        qolo_state_pub_node,
        rviz_node,
    ]

    return LaunchDescription(nodes)
