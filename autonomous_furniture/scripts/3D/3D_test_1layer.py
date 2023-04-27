import numpy as np

import matplotlib.pyplot as plt
from scipy import rand
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from autonomous_furniture.dynamical_system_animation3D import DynamicalSystemAnimation3D

from autonomous_furniture.agent3D import Furniture3D

import argparse


def threeD_test(args=[]):
    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()

    # control_points for the table
    table_legs_control_points = np.array([[2, 1.5], [2, -1.5], [-2, 1.5], [-2, -1.5]])
    table_legs_positions = np.copy(table_legs_control_points)
    # control_points = np.array([[0.6, 0.0], [-0.6, 0.0]])

    # , orientation = 1.6) Goal of the CuboidXd
    # , orientation = 1.6) Goal of the CuboidXd
    table_reference_goal = ObjectPose(
        position=np.array([6, 2.75]), orientation=np.pi / 2
    )
    table_reference_start = ObjectPose(position=np.array([0, 2.75]), orientation=0)

    table_legs = []
    for i in range(4):
        table_leg_shape = CuboidXd(
            axes_length=[0.3, 0.3],
            center_position=table_reference_start.transform_position_from_relative(
                np.copy(table_legs_positions[i])
            ),
            margin_absolut=0.5,
            orientation=0,
            tail_effect=False,
        )
        table_legs.append(table_leg_shape)

    start2 = ObjectPose(position=np.array([5, 1.7]), orientation=np.pi / 2)
    goal2 = ObjectPose(position=np.array([1, 3.7]), orientation=np.pi / 2)
    table_shape2 = CuboidXd(
        axes_length=[1, 2],
        center_position=start2.position,
        margin_absolut=0.5,
        orientation=start2.orientation,
        tail_effect=False,
    )

    layer_0 = [
        Furniture3D(
            shape_list=table_legs,
            shape_positions=table_legs_positions,
            obstacle_environment=obstacle_environment,
            control_points=table_legs_control_points,
            starting_pose=table_reference_start,
            goal_pose=table_reference_goal,
            name="table_legs",
        ),
        # Furniture3D(
        #     shape_list=[table_shape2],
        #     shape_positions=np.array([[0.0, 0.0]]),
        #     obstacle_environment=obstacle_environment,
        #     control_points=np.array([[0, -0.5], [0, 0.5]]),
        #     starting_pose=start2,
        #     goal_pose=goal2,
        #     static=False,
        #     name="static",
        # ),
    ]

    my_animation = DynamicalSystemAnimation3D(
        it_max=1000,
        dt_simulation=0.02,
        dt_sleep=0.02,
        animation_name=args.name,
    )

    my_animation.setup(
        layer_list=[layer_0],
        x_lim=[-3, 10],
        y_lim=[-2, 7],
        version="v2",
        mini_drag="dragdist",
        safety_module=False,
        emergency_stop=False,
        figsize=(10, 7),
    )

    my_animation.run(save_animation=args.rec)
    # my_animation.logs(len(my_furniture))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rec", action="store", default=False, help="Record flag")
    parser.add_argument(
        "--name", action="store", default="recording", help="Name of the simulation"
    )
    args = parser.parse_args()

    plt.close("all")
    plt.ion()

    threeD_test(args)
