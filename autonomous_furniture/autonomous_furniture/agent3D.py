"""
Autonomous two-dimensional agents which navigate in unstructured environments.
"""
import warnings
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum, auto

from asyncio import get_running_loop

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import matplotlib.pyplot as plt

from vartools.dynamical_systems.linear import ConstantValue
from vartools.states import ObjectPose, ObjectTwist
from vartools.dynamical_systems import LinearSystem

# from dynamic_obstacle_avoidance.obstacles import Obstacle
# from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles.ellipse_xd import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers.obstacle_container import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

# from vartools.states


class ObjectType(Enum):
    TABLE = auto()
    QOLO = auto()
    CHAIR = auto()
    HOSPITAL_BED = auto()
    OTHER = auto()


class Furniture3D:
    def __init__(
        self,
        shape: ObstacleContainer,
        shape_positions: Optional[np.ndarray],
    ) -> None:
        super().__init__()