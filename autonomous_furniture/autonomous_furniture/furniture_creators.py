"""
Furniture Factories
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2023-02-28

from typing import Optional

import math

import numpy as np

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.agent import Furniture
from autonomous_furniture.agent import ObjectType
from autonomous_furniture.obstacle_container import GlobalObstacleContainer

from autonomous_furniture.agent3D import Furniture3D
from vartools.states import ObjectPose


def create_chair(
    start_pose: ObjectPose,
    goal_pose: Optional[ObjectPose] = None,
    name: str = "",
    margin_absolut: float = 0.5,
    # center_position: np.ndarray, goal_pose: Optional[np.ndarray] = None
) -> Furniture:
    if not len(name):
        name = f"obstacle{len(GlobalObstacleContainer())}"

    if goal_pose is None:
        goal_pose = start_pose

    # Small difference of control points to allow changing the orientation
    # TODO: somehow the re-orientation does not work
    # control_points = np.array([[-0.05, 0], [0.05, 0.0]])
    control_points = np.array([[0.0, -0.00], [0, 0.00]])

    shape_ = Cuboid(
        axes_length=np.array([0.7, 0.6]),
        center_position=start_pose.position,
        margin_absolut=margin_absolut,
        orientation=start_pose.orientation,
        tail_effect=False,
    )

    new_furniture = Furniture(
        shape=shape_,
        obstacle_environment=GlobalObstacleContainer(),
        control_points=control_points,
        goal_pose=goal_pose,
        priority_value=1.0,
        name=name,
        object_type=ObjectType.CHAIR,
    )

    return new_furniture


def create_table(
    start_pose: ObjectPose,
    goal_pose: Optional[ObjectPose] = None,
    name: str = "",
    margin_absolut: float = 0.5,
    # center_position: np.ndarray, goal_pose: Optional[np.ndarray] = None
) -> Furniture:
    if not len(name):
        name = f"obstacle{len(GlobalObstacleContainer.get())}"

    if goal_pose is None:
        goal_pose = start_pose

    control_points = np.array([[0.45, 0], [-0.45, 0]])

    shape_ = Cuboid(
        axes_length=np.array([1.8, 0.8]),
        center_position=start_pose.position,
        margin_absolut=margin_absolut,
        orientation=start_pose.orientation,
        tail_effect=False,
    )

    new_furniture = Furniture(
        shape=shape_,
        obstacle_environment=GlobalObstacleContainer.get(),
        control_points=control_points,
        goal_pose=goal_pose,
        priority_value=1.0,
        name=name,
        object_type=ObjectType.TABLE,
        symmetry=math.pi,
    )

    return new_furniture


def create_four_chair_arrangement(
    center_position: np.ndarray, orientation: float = 0.0
):
    if orientation:
        raise NotImplementedError()

    delta_x = 0.5
    delta_y = 0.8
    agent_list: list[BaseAgent] = []

    new_furniture = create_table(
        start_pose=ObjectPose(position=center_position, orientation=0)
    )
    agent_list.append(new_furniture)

    new_furniture = create_chair(
        start_pose=ObjectPose(
            position=center_position + np.array([-delta_x, -delta_y]),
            orientation=math.pi / 2.0,
        )
    )
    agent_list.append(new_furniture)

    new_furniture = create_chair(
        start_pose=ObjectPose(
            position=center_position + np.array([delta_x, -delta_y]),
            orientation=math.pi / 2.0,
        ),
    )
    agent_list.append(new_furniture)

    new_furniture = create_chair(
        start_pose=ObjectPose(
            position=center_position + np.array([-delta_x, delta_y]),
            orientation=-math.pi / 2.0,
        ),
    )
    agent_list.append(new_furniture)

    new_furniture = create_chair(
        start_pose=ObjectPose(
            position=center_position + np.array([delta_x, delta_y]),
            orientation=-math.pi / 2.0,
        ),
    )
    agent_list.append(new_furniture)

    return agent_list


def add_walls(
    x_lim, y_lim, agent_list, wall_width=0.5, area_enlargement=1.5, wall_margin=0.5
) -> Furniture:
    # adds walls to the simulation
    total_displacement_center = wall_width / 2 + area_enlargement
    center_left = [x_lim[0] - total_displacement_center, np.average(y_lim)]
    center_down = [np.average(x_lim), y_lim[0] - total_displacement_center]
    center_right = [x_lim[1] + total_displacement_center, np.average(y_lim)]
    center_up = [np.average(x_lim), y_lim[1] + total_displacement_center]
    center_array = [center_left, center_down, center_right, center_up]
    orientation_array = [np.pi / 2, 0.0, np.pi / 2, 0.0]
    horizontal_wall_axis = np.array(
        [x_lim[1] - x_lim[0] + total_displacement_center * 2, wall_width]
    )
    vertical_wall_axis = np.array(
        [y_lim[1] - y_lim[0] + total_displacement_center * 2, wall_width]
    )
    axis_array = [
        vertical_wall_axis,
        horizontal_wall_axis,
        vertical_wall_axis,
        horizontal_wall_axis,
    ]

    for i in range(4):
        wall_pose = ObjectPose(
            position=center_array[i], orientation=orientation_array[i]
        )

        wall_shape = Cuboid(
            axes_length=axis_array[i],
            center_position=wall_pose.position,
            margin_absolut=wall_margin,
            orientation=wall_pose.orientation,
            tail_effect=False,
        )
        control_points = np.array(
            [[-axis_array[i][0] / 2, 0.0], [axis_array[i][0] / 2, 0.0]]
        )
        wall = Furniture(
            shape=wall_shape,
            obstacle_environment=GlobalObstacleContainer.get(),
            control_points=control_points,
            goal_pose=wall_pose,
            priority_value=1.0,
            name="wall",
            object_type=ObjectType.OTHER,
            symmetry=math.pi,
            static=True,
        )

        agent_list.append(wall)

    return agent_list


def create_hospital_bed(
    start_pose: ObjectPose,
    goal_pose: Optional[ObjectPose] = None,
    name: str = "",
    margin_absolut: float = 0.7,
    axes_length=np.array([2.0, 1.0])
    # center_position: np.ndarray, goal_pose: Optional[np.ndarray] = None
) -> Furniture:
    if not len(name):
        name = f"obstacle{len(GlobalObstacleContainer.get())}"

    if goal_pose is None:
        goal_pose = start_pose

    control_points = np.array([[0.5, 0], [-0.5, 0]])

    table_shape = Cuboid(
        axes_length=axes_length,
        center_position=start_pose.position,
        margin_absolut=margin_absolut,
        orientation=start_pose.orientation,
        tail_effect=False,
    )

    new_bed = Furniture(
        shape=table_shape,
        obstacle_environment=GlobalObstacleContainer.get(),
        control_points=control_points,
        goal_pose=goal_pose,
        priority_value=1.0,
        name=name,
        object_type=ObjectType.HOSPITAL_BED,
    )

    return new_bed


def set_goals_to_arrange_on_the_right(
    agent_list: list[BaseAgent],
    x_lim: list[float],
    y_lim: list[float],
    delta_x: float,
    delta_y: float,
    orientation: float = 0.0,
    arranging_type: ObjectType = ObjectType.TABLE,
):
    """Orderly rearange on the right - currently no 'finding of the closest position'."""
    num_y = math.floor((y_lim[1] - y_lim[0]) / delta_y)

    it_table = 0
    for agent in agent_list:
        if agent.object_type is not arranging_type:
            continue
        n_x = math.floor(it_table / num_y)
        n_y = it_table % num_y
        goal_position = np.array(
            [x_lim[-1] - (n_x + 0.5) * delta_x, y_lim[0] + (n_y + 0.5) * delta_y]
        )
        agent.set_goal_pose(ObjectPose(goal_position, orientation))

        it_table += 1


def set_goals_to_arrange_on_the_left(
    agent_list: list[BaseAgent],
    x_lim: list[float],
    y_lim: list[float],
    delta_x: float,
    delta_y: float,
    orientation: float = 0.0,
    arranging_type: ObjectType = ObjectType.CHAIR,
):
    """Orderly rearange on the right - currently no 'finding of the closest position'."""
    num_y = math.floor((y_lim[1] - y_lim[0]) / delta_y)

    it_table = 0
    for agent in agent_list:
        if agent.object_type is not arranging_type:
            continue
        n_x = math.floor(it_table / num_y)
        n_y = it_table % num_y
        goal_position = np.array(
            [x_lim[0] + (n_x + 0.5) * delta_x, y_lim[0] + (n_y + 0.5) * delta_y]
        )
        agent.set_goal_pose(ObjectPose(goal_position, orientation))

        it_table += 1

def create_standard_3D_chair_surface_back(obstacle_environment_lower, obstacle_environment_upper, start_pose: ObjectPose, goal_pose: ObjectPose, margins):
    ### CREATE CHAIR SECTIONS FOR ALL THE LAYERS
    chair_reference_start = ObjectPose(position=start_pose.position, orientation=start_pose.orientation)
    chair_reference_goal = ObjectPose(position=goal_pose.position, orientation=goal_pose.orientation)
    # lower layer
    chair_surface_control_points = np.array([[0.15, 0.15], [-0.15, 0.15], [0.15, -0.15], [-0.15, -0.15]])
    chair_surface_positions = np.array([[0.0, 0.0]])
    chair_surface_shape = Cuboid(
        axes_length=[0.5, 0.5],
        center_position=chair_reference_start.transform_position_from_relative(np.copy(chair_surface_positions[0])),
        margin_absolut=margins,
        orientation=chair_reference_start.orientation,
    )
    chair_surface_agent = Furniture3D(
        shape_list=[chair_surface_shape],
        shape_positions=chair_surface_positions,
        obstacle_environment=obstacle_environment_lower,
        control_points=chair_surface_control_points,
        starting_pose = ObjectPose(position=chair_reference_start.position, orientation=chair_reference_start.orientation),
        goal_pose=chair_reference_goal,
        name="chair_surface",
    )
    # upper layer
    chair_back_control_points = np.array([[0.25, 0.125], [0.25, -0.125]])
    chair_back_positions = np.array([[0.25, 0.0]])
    chair_back_shape = Cuboid(
        axes_length=[0.1,0.5],
        center_position=chair_reference_start.transform_position_from_relative(np.copy(chair_back_positions[0])),
        margin_absolut=margins,
        orientation=chair_reference_start.orientation,
    )
    chair_back_agent = Furniture3D(
        shape_list=[chair_back_shape],
        shape_positions=chair_back_positions,
        obstacle_environment=obstacle_environment_upper,
        control_points=chair_back_control_points,
        starting_pose = ObjectPose(position=chair_reference_start.position, orientation=chair_reference_start.orientation),
        goal_pose=chair_reference_goal,
        name="chair_back",
    )
    
    return chair_surface_agent, chair_back_agent

def create_standard_table_3D_surface_legs(obstacle_environment_lower, obstacle_environment_upper, start_pose: ObjectPose, goal_pose: ObjectPose, margins):
    ### CREATE TABLE SECTIONS FOR ALL THE LAYERS
    table_reference_goal = ObjectPose(position=goal_pose.position, orientation=goal_pose.orientation)
    table_reference_start = ObjectPose(position=start_pose.position, orientation=start_pose.orientation)
    
    # lower layer
    table_legs_control_points = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    table_legs_positions = np.copy(table_legs_control_points)
    table_legs_shapes = []
    for i in range(4):
        table_leg_shape = Cuboid(
            axes_length=[0.2, 0.2],
            center_position=table_reference_start.transform_position_from_relative(np.copy(table_legs_positions[i])),
            margin_absolut=margins,
            orientation=table_reference_start.orientation,
            tail_effect=False,
        )
        table_legs_shapes.append(table_leg_shape)
    table_legs_agent = Furniture3D(
        shape_list=table_legs_shapes,
        shape_positions=table_legs_positions,
        obstacle_environment=obstacle_environment_lower,
        control_points=table_legs_control_points,
        starting_pose = ObjectPose(position=table_reference_start.position, orientation=table_reference_start.orientation),
        goal_pose=table_reference_goal,
        name="table_legs",
    )
    # upper layer
    table_surface_control_points = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1], [1,0], [-1,0], [0, 1], [0, -1]])
    table_surface_positions = np.array([[0.0, 0.0]])
    table_surface_shape = Cuboid(
        axes_length=[2.2, 2.2],
        center_position=table_reference_start.transform_position_from_relative(np.copy(table_surface_positions[0])),
        margin_absolut=margins,
        orientation=table_reference_start.orientation,
    )
    table_surface_agent = Furniture3D(
        shape_list=[table_surface_shape],
        shape_positions=table_surface_positions,
        obstacle_environment=obstacle_environment_upper,
        control_points=table_surface_control_points,
        starting_pose = ObjectPose(position=table_reference_start.position, orientation=table_reference_start.orientation),
        goal_pose=table_reference_goal,
        name="table_surface",
    )
    return table_legs_agent, table_surface_agent