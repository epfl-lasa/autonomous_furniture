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

    table_shape = Cuboid(
        axes_length=np.array([0.7, 0.6]),
        center_position=start_pose.position,
        margin_absolut=margin_absolut,
        orientation=start_pose.orientation,
        tail_effect=False,
    )

    new_furniture = Furniture(
        shape=table_shape,
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

    table_shape = Cuboid(
        axes_length=np.array([1.8, 0.8]),
        center_position=start_pose.position,
        margin_absolut=margin_absolut,
        orientation=start_pose.orientation,
        tail_effect=False,
    )

    new_furniture = Furniture(
        shape=table_shape,
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


def create_hospital_bed(
    start_pose: ObjectPose,
    goal_pose: Optional[ObjectPose] = None,
    name: str = "",
    margin_absolut: float = 0.7
    # center_position: np.ndarray, goal_pose: Optional[np.ndarray] = None
) -> Furniture:
    if not len(name):
        name = f"obstacle{len(GlobalObstacleContainer.get())}"

    if goal_pose is None:
        goal_pose = start_pose

    control_points = np.array([[0.5, 0], [-0.5, 0]])

    table_shape = Cuboid(
        axes_length=np.array([2.0, 1.0]),
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