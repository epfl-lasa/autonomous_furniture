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
    create_3D_table_surface_legs,
    create_3D_chair,
)

import argparse

import pathlib


def threeD_test(args=[]):
    parameter_file = (
        str(pathlib.Path(__file__).parent.resolve())
        + "/parameters/experiment_simulation.yaml"
    )

    # List of environment shared by all the furniture/agent in the same layer
    obstacle_environment_lower = ObstacleContainer()
    obstacle_environment_upper = ObstacleContainer()

    margins = 0.05

    ### CREATE STATIC TABLE SECTIONS FOR ALL THE LAYERS
    static_table_reference_start = ObjectPose(
        position=np.array([3.0, 1.0]), orientation=np.pi / 2
    )
    static_table_reference_goal = ObjectPose(
        position=np.array([-1.0, 4.0]), orientation=np.pi / 2
    )

    [
        static_table_legs_agent,
        static_table_surface_agent,
    ] = create_3D_table_surface_legs(
        obstacle_environment_legs=obstacle_environment_lower,
        obstacle_environment_surface=obstacle_environment_upper,
        start_pose=static_table_reference_start,
        goal_pose=static_table_reference_goal,
        margin_shape=margins,
        margin_control_points=0.1,
        axes_table=[1.6, 0.8],
        axes_legs=[0.04, 0.04],
        ctr_points_number=[2, 2],
        static=True,
        parameter_file=parameter_file,
    )

    ### CREATE MOBILE LOW TABLE SECTIONS FOR ALL THE LAYERS
    mobile_table_reference_start = ObjectPose(
        position=np.array([2.25, 1.25]), orientation=0.0
    )
    mobile_table_reference_goal = ObjectPose(
        position=np.array([4.5, 0.5]), orientation=0.0
    )

    [
        mobile_table_legs_agent,
        mobile_table_surface_agent,
    ] = create_3D_table_surface_legs(
        obstacle_environment_legs=None,
        obstacle_environment_surface=obstacle_environment_lower,
        start_pose=mobile_table_reference_start,
        goal_pose=mobile_table_reference_goal,
        margin_shape=margins,
        margin_control_points=0.0,
        axes_table=[0.55, 0.55],
        axes_legs=[0.04, 0.04],
        ctr_points_number=[4, 4],
        static=False,
        parameter_file=parameter_file,
    )
    mobile_table_surface_agent.safety_module = False

    chair_left_reference_start = ObjectPose(
        position=np.array([1.5, 0.5]), orientation=-np.pi / 2
    )
    chair_left_reference_goal = ObjectPose(
        position=np.array([2.6, 1.0]), orientation=np.pi / 2
    )

    [
        chair_left_surface_agent,
        chair_left_back_agent,
    ] = create_3D_chair(
        obstacle_environment_surface=obstacle_environment_lower,
        obstacle_environment_back=obstacle_environment_upper,
        start_pose=chair_left_reference_start,
        goal_pose=chair_left_reference_goal,
        margin_absolut=margins,
        margin_ctr_pt=0.0,
        back_axis=[0.37, 0.035],
        back_ctr_pt_number=[4, 2],
        back_positions=np.array([[0.0, 0.2]]),
        surface_axis=[0.4, 0.4],
        surface_ctr_pt_number=[3, 3],
        surface_positions=np.array([[0.0, 0.0]]),
        parameter_file=parameter_file,
    )

    chair_right_reference_start = ObjectPose(
        position=np.array([4.75, 1.75]), orientation=np.pi / 2
    )
    chair_right_reference_goal = ObjectPose(
        position=np.array([3.4, 1.0]), orientation=-np.pi / 2
    )

    [
        chair_right_surface_agent,
        chair_right_back_agent,
    ] = create_3D_chair(
        obstacle_environment_surface=obstacle_environment_lower,
        obstacle_environment_back=obstacle_environment_upper,
        start_pose=chair_right_reference_start,
        goal_pose=chair_right_reference_goal,
        margin_absolut=margins,
        margin_ctr_pt=0.0,
        back_axis=[0.37, 0.035],
        back_ctr_pt_number=[4, 2],
        back_positions=np.array([[0.0, 0.2]]),
        surface_axis=[0.4, 0.4],
        surface_ctr_pt_number=[3, 3],
        surface_positions=np.array([[0.0, 0.0]]),
        parameter_file=parameter_file,
    )

    layer_lower = [
        chair_left_surface_agent,
        chair_right_surface_agent,
        mobile_table_surface_agent,
        static_table_legs_agent,
    ]
    layer_upper = [
        chair_left_back_agent,
        chair_right_back_agent,
        None,
        static_table_surface_agent,
    ]

    # layer_lower = [mobile_table_surface_agent, static_table_legs_agent]
    # layer_upper = [None, static_table_surface_agent]

    # layer_lower = [chair_left_surface_agent, static_table_legs_agent]
    # layer_upper = [chair_left_back_agent, static_table_surface_agent]

    my_animation = DynamicalSystemAnimation3D(parameter_file=parameter_file)

    my_animation.setup(
        layer_list=[layer_lower, layer_upper], parameter_file=parameter_file
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
