from typing import Optional
import time

import math
from math import sin, cos, pi

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from tf2_ros import TransformBroadcaster, TransformStamped

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.avoidance.dynamic_crowd_avoider import (
    obstacle_environment_slicer,
)

from autonomous_furniture.analysis.calc_time import calculate_relative_position
from autonomous_furniture.analysis.calc_time import relative2global
from autonomous_furniture.analysis.calc_time import global2relative
from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation

from autonomous_furniture.agent import Furniture, Person
from autonomous_furniture.attractor_dynamics import AttractorDynamics
from autonomous_furniture.message_generation import euler_to_quaternion


class RvizPublisher(Node):
    def __init__(self):
        pass


class RvizSimulator:
    def __init__(
        self, it_max: int, dt_sleep: float = 0.05, dt_simulation: float = 0.05
    ) -> None:

        pass

    def update_step(self, ii: int) -> None:
        pass


class GlobalObstacleContainer:
    """Singleton-Class of GlobalFurnitureContainer."""

    # _instance: Optional[GlobalFurnitureContainer] = None # Gives me an error (!?)
    _instance: Optional[list] = None
    # _furniture_list: list[Furniture] = []
    # _obstacle_container: ObstacleContainer = ObstacleContainer()
    _obstacle_list: list[Obstacle] = []

    def __new__(cls, *args, **kwargs):
        print("Create instance of GlobalFurnitureContainer.")
        cls._instance = super(GlobalObstacleContainer, cls).__new__(
            cls, *args, **kwargs
        )
        return cls._instance

    def __init__(self):
        if self._instance is None:
            GlobalFurnitureContainer.__new__()

    def append(self, furniture: Furniture) -> None:
        self._obstacle_list.append(furniture)
        # self._obstacle_container.append(furniture.get_obstacle_shape())

    def __getitem__(self, key: int) -> Obstacle:
        return self._obstacle_list[key]

    def __setitem__(self, key: int, value: Obstacle) -> None:
        self._obstacle_list[key] = value

    # def get_obstacle_list(self) -> ObstacleContainer:
    #     return self._obstacle_container


def create_hospital_bed(
    center_position: np.ndarray, goal_pose: Optional[np.ndarray] = None
) -> Furniture:
    control_points = np.array([[0.6, 0], [-0.6, 0]])
    goal = ObjectPose(position=np.array([2.5, 3]), orientation=0)

    table_shape = Cuboid(
        axes_length=np.array([2.0, 1]),
        center_position=center_position,
        margin_absolut=1,
        orientation=0,
        tail_effect=False,
    )

    new_bed = Furniture(
        shape=table_shape,
        obstacle_environment=GlobalFurnitureContainer(),
        control_points=control_points,
        goal_pose=goal,
        priority_value=1,
        name="hospital_bed",
    )

    return new_bed


def main_animation():
    axes_length = [2.4, 1.1]

    global_furniture = GlobalFurnitureContainer()
    new_bed = create_hospital_bed(center_position=[3.0, 4.0])
    breakpoint()

    global_furniture.append(new_bed)

    my_animation = DynamicalSystemAnimation(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name="furniture_animation",
    )

    my_animation.setup(
        obstacle_environment=global_furniture.get_obstacle_list(),
        agent=global_furniture,
        x_lim=[-3, 8],
        y_lim=[-2, 7],
        version="v2",
        mini_drag="dragdist",
    )

    my_animation.run(save_animation="False")
    my_animation.logs(len(my_furniture))


if __name__ == "__main__":
    plt.close("all")
    main_animation()
