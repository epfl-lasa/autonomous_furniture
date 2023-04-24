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

from autonomous_furniture.furniture_creators import (
    create_standard_table_3D_surface_legs,
    create_standard_3D_chair_surface_back,
)

import argparse


def threeD_test(args=[]):
    # List of environment shared by all the furniture/agent in the same layer
    obstacle_environment_lower = ObstacleContainer()
    obstacle_environment_upper = ObstacleContainer()

    ### CREATE TABLE SECTIONS FOR ALL THE LAYERS
    table_reference_goal = ObjectPose(position=np.array([4, 2.75]), orientation=0)
    table_reference_start = ObjectPose(position=np.array([0, 2.75]), orientation=0)

    table_legs_agent, table_surface_agent = create_standard_table_3D_surface_legs(
        obstacle_environment_lower,
        obstacle_environment_upper,
        table_reference_start,
        table_reference_goal,
        margins=0.1,
    )

    chair_down_reference_start = ObjectPose(
        position=np.array([4, 0]), orientation=-np.pi / 2
    )
    chair_down_reference_goal = ObjectPose(
        position=np.array([4, 1.7]), orientation=-np.pi / 2
    )

    (
        chair_down_surface_agent,
        chair_down_back_agent,
    ) = create_standard_3D_chair_surface_back(
        obstacle_environment_lower,
        obstacle_environment_upper,
        chair_down_reference_start,
        chair_down_reference_goal,
        margins=0.1,
    )

    chair_up_reference_start = ObjectPose(
        position=np.array([4, 6]), orientation=np.pi / 2
    )
    chair_up_reference_goal = ObjectPose(
        position=np.array([4, 3.7]), orientation=np.pi / 2
    )

    chair_up_surface_agent, chair_up_back_agent = create_standard_3D_chair_surface_back(
        obstacle_environment_lower,
        obstacle_environment_upper,
        chair_up_reference_start,
        chair_up_reference_goal,
        margins=0.1,
    )

    # chair_surface_agent.priority = 1e3
    # chair_back_agent.priority = 1e3

    layer_lower = [table_legs_agent, chair_down_surface_agent]
    layer_upper = [table_surface_agent, chair_down_back_agent]

    my_animation = DynamicalSystemAnimation3D(
        it_max=1000,
        dt_simulation=0.02,
        dt_sleep=0.02,
        animation_name=args.name,
    )

    my_animation.setup(
        obstacle_environment_list=[
            obstacle_environment_lower,
            obstacle_environment_upper,
        ],
        layer_list=[layer_lower, layer_upper],
        x_lim=[-2, 6],
        y_lim=[-1, 5],
        version="v1",
        mini_drag="nodrag",
        safety_module=True,
        emergency_stop=True,
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
