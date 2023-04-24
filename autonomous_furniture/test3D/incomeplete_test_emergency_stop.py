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
    # List of environment shared by all the furniture/agent in the same layer
    obstacle_environment_lower = ObstacleContainer()
    obstacle_environment_upper = ObstacleContainer()

    ### CREATE TABLE SECTIONS FOR ALL THE LAYERS
    table_reference_goal = ObjectPose(position=np.array([6, 2.75]), orientation=0)
    table_reference_start = ObjectPose(position=np.array([0, 2.75]), orientation=0)

    # lower layer
    table_legs_control_points = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    table_legs_positions = np.copy(table_legs_control_points)
    table_legs_shapes = []
    for i in range(4):
        table_leg_shape = CuboidXd(
            axes_length=[0.2, 0.2],
            center_position=table_reference_start.transform_position_from_relative(
                np.copy(table_legs_positions[i])
            ),
            margin_absolut=0.5,
            orientation=table_reference_start.orientation,
            tail_effect=False,
        )
        table_legs_shapes.append(table_leg_shape)
    table_legs_agent = Furniture3D(
        shape_list=table_legs_shapes,
        shape_positions=table_legs_positions,
        obstacle_environment=obstacle_environment_lower,
        control_points=table_legs_control_points,
        starting_pose=ObjectPose(
            position=table_reference_start.position,
            orientation=table_reference_start.orientation,
        ),
        goal_pose=table_reference_goal,
        name="table_legs",
    )
    # upper layer
    table_surface_control_points = np.array(
        [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]]
    )
    table_surface_positions = np.array([[0.0, 0.0]])
    table_surface_shape = CuboidXd(
        axes_length=[2.2, 2.2],
        center_position=table_reference_start.transform_position_from_relative(
            np.copy(table_surface_positions[0])
        ),
        margin_absolut=0.5,
        orientation=table_reference_start.orientation,
    )
    table_surface_agent = Furniture3D(
        shape_list=[table_surface_shape],
        shape_positions=table_surface_positions,
        obstacle_environment=obstacle_environment_upper,
        control_points=table_surface_control_points,
        starting_pose=ObjectPose(
            position=table_reference_start.position,
            orientation=table_reference_start.orientation,
        ),
        goal_pose=table_reference_goal,
        name="table_surface",
    )
    ### CREATE CHAIR SECTIONS FOR ALL THE LAYERS
    chair_reference_start = ObjectPose(position=np.array([5, 1.7]), orientation=0)
    chair_reference_goal = ObjectPose(position=np.array([1, 3.7]), orientation=0)
    # lower layer
    chair_surface_control_points = np.array(
        [[0.15, 0.15], [-0.15, 0.15], [0.15, -0.15], [-0.15, -0.15]]
    )
    chair_surface_positions = np.array([[0.0, 0.0]])
    chair_surface_shape = CuboidXd(
        axes_length=[0.5, 0.5],
        center_position=chair_reference_start.transform_position_from_relative(
            np.copy(chair_surface_positions[0])
        ),
        margin_absolut=0.5,
        orientation=chair_reference_start.orientation,
    )
    chair_surface_agent = Furniture3D(
        shape_list=[chair_surface_shape],
        shape_positions=chair_surface_positions,
        obstacle_environment=obstacle_environment_lower,
        control_points=chair_surface_control_points,
        starting_pose=ObjectPose(
            position=chair_reference_start.position,
            orientation=chair_reference_start.orientation,
        ),
        goal_pose=chair_reference_goal,
        name="chair_surface",
    )
    # upper layer
    chair_back_control_points = np.array([[0.25, 0.125], [0.25, -0.125]])
    chair_back_positions = np.array([[0.25, 0.0]])
    chair_back_shape = CuboidXd(
        axes_length=[0.1, 0.5],
        center_position=chair_reference_start.transform_position_from_relative(
            np.copy(chair_back_positions[0])
        ),
        margin_absolut=0.5,
        orientation=chair_reference_start.orientation,
    )
    chair_back_agent = Furniture3D(
        shape_list=[chair_back_shape],
        shape_positions=chair_back_positions,
        obstacle_environment=obstacle_environment_upper,
        control_points=chair_back_control_points,
        starting_pose=ObjectPose(
            position=chair_reference_start.position,
            orientation=chair_reference_start.orientation,
        ),
        goal_pose=chair_reference_goal,
        name="chair_back",
    )

    layer_lower = [table_legs_agent, chair_surface_agent]
    layer_upper = [table_surface_agent, chair_back_agent]

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
        x_lim=[-3, 10],
        y_lim=[-2, 7],
        version="v1",
        mini_drag="nodrag",
        safety_module=True,
        emergency_stop=True,
        figsize=(10, 7),
    )

    while chair_surface_agent.stop == False:
        my_animation.update_step(anim=False)

    assert (
        chair_back_agent.stop == True
    ), "Expect to stop all agent sections once one section has emergency stop triggered!"
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
