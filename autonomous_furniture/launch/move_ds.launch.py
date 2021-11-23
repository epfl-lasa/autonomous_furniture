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
        ]
    )
    chair2_description = {"use_sim_time": use_sim_time, "robot_description": chair2_description_content}

    wheelchair_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("objects_descriptions"), "urdf/wheelchair.urdf.xacro"]),
            " ",
            "prefix:=wheelchair_ ",
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
        ]
    )
    qolo_description = {"use_sim_time": use_sim_time, "robot_description": qolo_description_content}

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

    qolo_state_pub_node = Node(
        package="robot_state_publisher",
        namespace="qolo",
        executable="robot_state_publisher",
        output="both",
        parameters=[qolo_description],
    )

    table_pos_pub_node = Node(
        package='autonomous_furniture',
        executable='table_state_publisher',
        name='table_state_publisher',
        output='screen',
    )

    chair1_pos_pub_node = Node(
        package='autonomous_furniture',
        executable='chair_state_publisher',
        name='chair_state_publisher',
        output='screen',
        parameters=[
            {"prefix": "chair_1_"}
        ],
    )

    chair2_pos_pub_node = Node(
        package='autonomous_furniture',
        executable='state_publisher',
        name='state_publisher',
        output='screen',
        parameters=[
            {"prefix": "chair_2_"}
        ],
    )

    qolo_pos_pub_node = Node(
        package='autonomous_furniture',
        executable='qolo_state_publisher',
        name='qolo_state_publisher',
        output='screen',
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("objects_descriptions"), "rviz/move_multiple_ds.rviz"])],
        output="log",
    )

    sim_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    ds_algo = Node(
        package="autonomous_furniture",
        executable="DS_state_publisher",
        name="DS_state_publisher",
        output="screen",
    )

    nodes = [
        sim_arg,
        table_state_pub_node,
        # table_pos_pub_node,
        # chair1_state_pub_node,
        # chair1_pos_pub_node,
        # chair2_state_pub_node,
        # chair2_pos_pub_node,
        qolo_state_pub_node,
        # qolo_pos_pub_node,
        rviz_node,
        ds_algo,
    ]

    return LaunchDescription(nodes)
