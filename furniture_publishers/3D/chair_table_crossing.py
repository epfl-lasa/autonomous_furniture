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
from autonomous_furniture.rviz_animator3D import RvizSimulator3D
from autonomous_furniture.furniture_creators import (
    create_standard_table_3D_surface_legs,
    create_standard_3D_chair_surface_back,
)

import argparse
import rclpy
from rclpy.node import Node


def matplotlib_simulation(args=[]):
    my_animation = DynamicalSystemAnimation3D(
        it_max=1000,
        dt_simulation=0.03,
        dt_sleep=0.03,
        animation_name=args.name,
    )

    my_animation.setup(
        layer_list=create_layer_list(),
        x_lim=[-2, 3],
        y_lim=[-2, 3],
        version="v2",
        mini_drag="nodrag",
        safety_module=True,
        emergency_stop=True,
        figsize=(10, 7),
    )

    my_animation.run(save_animation=args.rec)


def rviz_simulation():
    print("Starting publishing node")
    rclpy.init()

    my_animation = RvizSimulator3D(
        it_max=1000,
        dt_simulation=0.01,
        dt_sleep=0.01,
    )

    my_animation.setup(
        layer_list=create_layer_list(),
        version="v2",
        mini_drag="nodrag",
        safety_module=True,
        emergency_stop=True,
    )

    rclpy.spin(my_animation)

    try:
        rckpy.shutdown()
    except:
        breakpoint()


def create_layer_list():
    # List of environment shared by all the furniture/agent in the same layer
    obstacle_environment_lower = ObstacleContainer()
    obstacle_environment_upper = ObstacleContainer()

    ### CREATE TABLE SECTIONS FOR ALL THE LAYERS
    table_reference_start = ObjectPose(position=np.array([0.0, 1.0]), orientation=0)
    table_reference_goal = ObjectPose(position=np.array([1.0, 1.0]), orientation=0)

    table_legs_agent, table_surface_agent = create_standard_table_3D_surface_legs(
        obstacle_environment_lower,
        obstacle_environment_upper,
        table_reference_start,
        table_reference_goal,
        margins=0.1,
        axes_legs=[0.05, 0.2],
        static=True,
    )

    chair_moving_reference_start = ObjectPose(
        position=np.array([0.8, 0]), orientation=np.pi / 2
    )
    chair_moving_reference_goal = ObjectPose(
        position=np.array([-1.5, 1.0]), orientation=np.pi / 2
    )

    (
        chair_moving_surface_agent,
        chair_moving_back_agent,
    ) = create_standard_3D_chair_surface_back(
        obstacle_environment_lower,
        obstacle_environment_upper,
        chair_moving_reference_start,
        chair_moving_reference_goal,
        margins=0.05,
    )

    chair_static_reference_start = ObjectPose(
        position=np.array([0.0, -0.1]), orientation=np.pi / 2
    )
    chair_static_reference_goal = ObjectPose(
        position=np.array([0.0, 1.0]), orientation=np.pi / 2
    )

    (
        chair_static_surface_agent,
        chair_static_back_agent,
    ) = create_standard_3D_chair_surface_back(
        obstacle_environment_lower,
        obstacle_environment_upper,
        chair_static_reference_start,
        chair_static_reference_goal,
        margins=0.05,
        static=True,
    )

    chair_moving_surface_agent.name = "chair_moving"
    chair_static_surface_agent.name = "chair_static"
    chair_moving_back_agent.name = "chair_moving"
    chair_static_back_agent.name = "chair_static"

    # table_surface_agent.priority = 1e6
    # table_legs_agent.priority = 1e6

    layer_lower = [
        table_legs_agent,
        chair_static_surface_agent,
        chair_moving_surface_agent,
    ]
    layer_upper = [
        table_surface_agent,
        chair_static_back_agent,
        chair_moving_back_agent,
    ]
    return [layer_lower, layer_upper]

    # chair_moving_surface_agent._obstacle_environment.append(
    #     table_surface_agent._shape_list[0]
    # )

    # layer = [
    #     table_surface_agent,
    #     chair_static_surface_agent,
    #     chair_moving_surface_agent,
    # ]
    # return [layer]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rec", action="store", default=False, help="Record flag")
    parser.add_argument(
        "--name", action="store", default="recording", help="Name of the simulation"
    )
    args = parser.parse_args()

    plt.close("all")
    plt.ion()

    matplotlib_simulation(args)
    # rviz_simulation()
