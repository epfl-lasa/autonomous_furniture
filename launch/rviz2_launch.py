import os
import sys

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory('autonomous_furniture'))
    path_u = os.path.join(base_path, 'chair.urdf')
    urdf = open(path_u).read()

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("object_descriptions"), "objects/open_box.urdf.xacro"]),
            " ",
            "prefix:=box_ ",
            "connected_to:='' ",
            "xyz:='1 0.5 0.25' ",
            "rpy:='1.57 0 0' ",
            "height:=0.15 ",
            "x_size:=0.5 ",
            "y_size:=0.5"
        ]
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher', name='robot_state_publisher', executable='robot_state_publisher',
            parameters=[{'robot_description': urdf}],
        ),
        Node(
            package='rviz2', name='rviz2', executable='rviz2', arguments=['-d', str('~/.rviz2/default.rviz')],

        )
    ])


def main(argv=sys.argv[1:]):
    """Run lifecycle nodes via launch."""
    ld = generate_launch_description()
    ls = launch.LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    main()

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
