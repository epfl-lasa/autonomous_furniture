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

from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.agent import Furniture, Person
from autonomous_furniture.attractor_dynamics import AttractorDynamics
from autonomous_furniture.message_generation import euler_to_quaternion


class RvizPublisher(Node):
    def __init__(self):
        pass


from enum import Enum, auto


class ObjectType(Enum):
    TABLE = auto()
    QOLO = auto()
    CHAIR = auto()
    HOSPITAL_BED = auto()
    OTHER = auto()


# @dataclass
class RvizSimulator(Node):
    # it_max: int
    # dt_sleep: float = 0.05
    # dt_simulation: float = 0.05
    def __init__(
        self, it_max: int, dt_sleep: float = 0.05, dt_simulation: float = 0.05
    ) -> None:
        self.animation_paused = False

        self.it_max = it_max
        self.dt_sleep = dt_sleep
        self.dt_simulation = dt_simulation

        rclpy.init()
        super().__init__("furniture_publisher")

        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.node_name = self.get_name()
        self.odom_transforme = TransformStamped()
        self.odom_transformer.header.frame_id = "world"

    def update_state_publisher(
        self,
        pose,
        frame_prefix: str,
        object_type: ObjectType = ObjectType.OTHER,
    ):
        self.odom_trans.child_frame_id = frame_prefix + "/" + "base_link"

        now = self.get_clock().now()
        self.odom_trans.header.stamp = now.to_msg()
        self.odom_trans.transform.translation.x = pose.position[0]
        self.odom_trans.transform.translation.y = pose.position[1]
        self.odom_trans.transform.translation.z = 0.0
        # Roll-Pitch-Yaw
        self.odom_trans.transform.rotation = euler_to_quaternion(0, 0, pose.rotation)

        if object_type is ObjectType.TABLE:
            self.odom_trans.transform.translation.x = pose.position[0] + 0.2

        elif object_type is ObjectType.QOLO:
            self.odom_trans.transform.translation.z = 0.2

        # send the joint state and transform
        self.broadcaster.sendTransform(self.odom_trans)

    def on_click(self, event):
        self.animation_paused = not self.animation_paused

    def setup(
        self,
        obstacle_environment: ObstacleContainer,
        agent: list[BaseAgent],
    ):
        dim = 2
        self.number_agent = len(agent)

        self.position_list = np.zeros((dim, self.it_max))
        self.time_list = np.zeros((self.it_max))
        self.position_list = [agent[ii].position for ii in range(self.number_agent)]
        self.agent = agent

        self.agent_pos_saver = []
        for i in range(self.number_agent):
            self.agent_pos_saver.append([])
        for i in range(self.number_agent):
            self.agent_pos_saver[i].append(self.agent[i].position)

        self.obstacle_environment = obstacle_environment
        self.converged: bool = False  # If all the agent has converged

    def update_step(self, ii: int) -> None:
        for jj in range(self.number_agent):
            self.agent[jj].update_velocity(
                mini_drag="no_drag", version="v2", emergency_stop=True
            )
            self.agent[jj].compute_metrics(self.dt_simulation)
            self.agent[jj].do_velocity_step(self.dt_simulation)

        for obs in range(num_obs):
                if obs == num_obs - 1:
                    u_obs_vel = obs_vel / np.linalg.norm(obs_vel)
                    x_vec = np.array([1, 0])
                    dot_prod = np.dot(x_vec, u_obs_vel)
                    qolo_dir = np.arccos(dot_prod)
                    self.update_state_publisher(
                        obs_name[obs],
                        obstacle_environment[-1].center_position,
                        qolo_dir,
                    )
                else:
                    self.update_state_publisher(
                        obs_name[obs],
                        obstacle_environment[obs].center_position,
                        obstacle_environment[obs].orientation,


class GlobalObstacleContainer:
    """Singleton-Class of GlobalFurnitureContainer."""

    # _instance: Optional[GlobalFurnitureContainer] = None # Gives me an error (!?)
    _instance: Optional[list] = None
    # _furniture_list: list[Furniture] = []
    # _obstacle_container: ObstacleContainer = ObstacleContainer()
    _obstacle_list: list[Obstacle] = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("Create instance of GlobalFurnitureContainer.")
            cls._instance = super(GlobalObstacleContainer, cls).__new__(
                cls, *args, **kwargs
            )
        return cls._instance

    def __init__(self):
        pass

    def append(self, furniture: Furniture) -> None:
        self._obstacle_list.append(furniture)
        # self._obstacle_container.append(furniture.get_obstacle_shape())

    def __getitem__(self, key: int) -> Obstacle:
        return self._obstacle_list[key]

    def __setitem__(self, key: int, value: Obstacle) -> None:
        self._obstacle_list[key] = value

    def __len__(self) -> int:
        return len(self._obstacle_list)

    # def get_obstacle_list(self) -> ObstacleContainer:
    #     return self._obstacle_container


def create_hospital_bed(
    start_pose: ObjectPose,
    goal_pose: ObjectPose,
    name: str = "",
    # center_position: np.ndarray, goal_pose: Optional[np.ndarray] = None
) -> Furniture:
    if not len(name):
        name = f"obstacle{len(GlobalObstacleContainer())}"

    control_points = np.array([[0.6, 0], [-0.6, 0]])

    table_shape = Cuboid(
        axes_length=np.array([2.0, 1.0]),
        center_position=start_pose.position,
        margin_absolut=1.0,
        orientation=start_pose.orientation,
        tail_effect=False,
    )

    new_bed = Furniture(
        shape=table_shape,
        obstacle_environment=GlobalObstacleContainer(),
        control_points=control_points,
        goal_pose=goal_pose,
        priority_value=1.0,
        name="hospital_bed",
    )

    return new_bed


def main_animation():
    axes_length = [2.4, 1.1]

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


if __name__ == "__main__":
    plt.close("all")
    main_animation()
