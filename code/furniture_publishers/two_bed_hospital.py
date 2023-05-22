from typing import Optional

import math

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from autonomous_furniture.agent import ObjectType
from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.agent import Furniture, Person

from autonomous_furniture.obstacle_container import GlobalObstacleContainer
from autonomous_furniture.furniture_creators import create_four_chair_arrangement
from autonomous_furniture.furniture_creators import (
    set_goals_to_arrange_on_the_left,
    set_goals_to_arrange_on_the_right,
)
from autonomous_furniture.rviz_animator import RvizSimulator


def two_bed_animation():
    agent_list: list[BaseAgent] = []

    new_bed = create_hospital_bed(
        start_pose=ObjectPose(position=np.array([-3.0, 0.0]), orientation=0),
        goal_pose=ObjectPose(position=np.array([3.0, 4.0]), orientation=0),
    )
    agent_list.append(new_bed)

    new_bed = create_hospital_bed(
        start_pose=ObjectPose(position=np.array([-3.0, 2.0]), orientation=0),
        goal_pose=ObjectPose(position=np.array([3.0, 2.0]), orientation=0),
    )
    agent_list.append(new_bed)

    my_animation = DynamicalSystemAnimation(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name="furniture_animation",
    )

    my_animation.setup(
        obstacle_environment=GlobalObstacleContainer(),
        agent=agent_list,
        x_lim=[-3, 8],
        y_lim=[-2, 7],
        figsize=(5, 4),
        version="v2",
        mini_drag="dragdist",
    )

    # GlobalObstacleContainer()
    my_animation.run(save_animation=False)
    my_animation.logs(len(agent_list))
