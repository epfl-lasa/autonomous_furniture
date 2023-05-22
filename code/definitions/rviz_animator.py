from typing import Optional
import time

import math
from math import sin, cos, pi

import numpy as np
import matplotlib.pyplot as plt

import keyboard

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

from autonomous_furniture.agent import ObjectType
from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.agent import Furniture, Person

from autonomous_furniture.obstacle_container import GlobalObstacleContainer
from autonomous_furniture.furniture_creators import create_chair
from autonomous_furniture.furniture_creators import create_table
from autonomous_furniture.furniture_creators import create_hospital_bed
from autonomous_furniture.furniture_creators import create_four_chair_arrangement
from autonomous_furniture.furniture_creators import (
    set_goals_to_arrange_on_the_left,
    set_goals_to_arrange_on_the_right,
)

from autonomous_furniture.attractor_dynamics import AttractorDynamics
from autonomous_furniture.message_generation import euler_to_quaternion


class RvizSimulator(Node):
    # it_max: int
    # dt_sleep: float = 0.05
    # dt_simulation: float = 0.05
    def __init__(
        self, it_max: int, dt_sleep: float = 0.05, dt_simulation: float = 0.01
    ) -> None:
        self.animation_paused = False

        self.it_max = it_max
        self.dt_sleep = dt_sleep
        self.dt_simulation = dt_simulation

        self.period = self.dt_simulation

        # rclpy.init()
        super().__init__("furniture_publisher")

        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.node_name = self.get_name()
        self.odom_transformer = TransformStamped()
        self.odom_transformer.header.frame_id = "world"

        self.ii = 0
        self._pause = False
        self.timer = self.create_timer(self.period, self.update_step)

    def update_state_publisher(
        self,
        pose,
        frame_prefix: str,
        object_type: ObjectType = ObjectType.OTHER,
    ):
        self.odom_transformer.child_frame_id = frame_prefix + "/" + "base_link"

        self.odom_transformer.header.stamp = self.get_clock().now().to_msg()
        self.odom_transformer.transform.translation.x = pose.position[0]
        self.odom_transformer.transform.translation.y = pose.position[1]
        self.odom_transformer.transform.translation.z = 0.0
        # Roll-Pitch-Yaw

        if object_type is ObjectType.HOSPITAL_BED:
            self.odom_transformer.transform.rotation = euler_to_quaternion(
                math.pi / 2.0, 0, pose.orientation
            )
        else:
            self.odom_transformer.transform.rotation = euler_to_quaternion(
                0, 0, pose.orientation
            )

        if object_type is ObjectType.TABLE:
            self.odom_transformer.transform.translation.x = pose.position[0] + 0.2

        elif object_type is ObjectType.QOLO:
            self.odom_transformer.transform.translation.z = 0.2
            self.odom_transformer.transform.rotation = euler_to_quaternion(
                0, 0, pose.orientation
            )  # rpy

        # send the joint state and transform
        self.broadcaster.sendTransform(self.odom_transformer)

    def on_click(self, event):
        self.animation_paused = not self.animation_paused

    def setup(
        self,
        obstacle_environment: ObstacleContainer,
        agent: list[BaseAgent],
    ):
        self.dim = 2
        self.number_agent = len(agent)
        self.agent = agent

        # self.position_list = np.zeros((dim, self.it_max))
        # self.time_list = np.zeros((self.it_max))
        # self.position_list = [agent[ii].position for ii in range(self.number_agent)]

        # self.agent_pos_saver = []
        # for i in range(self.number_agent):
        #     self.agent_pos_saver.append([])
        # for i in range(self.number_agent):
        #     self.agent_pos_saver[i].append(self.agent[i].position)

        self.obstacle_environment = obstacle_environment
        self.converged: bool = False  # If all the agent has converged

    def update_step(self) -> None:
        self.ii += 1
        if not (self.ii % 20):
            print(f"Iteration {self.ii}.")

        for jj in range(self.number_agent):
            self.agent[jj].update_velocity(
                # mini_drag="nodrag",
                mini_drag="dragdist",
                version="v2",
                emergency_stop=True,
                time_step=self.dt_simulation,
            )
            # self.agent[jj].compute_metrics(self.dt_simulation)
            self.agent[jj].do_velocity_step(self.dt_simulation)

        for ii, agent in enumerate(self.agent):
            if agent.object_type == ObjectType.QOLO:
                # u_obs_vel = agent.linear_velocity / np.linalg.norm(agent.linear_velocity)
                # x_vec = np.array([1, 0])
                # dot_prod = np.dot(x_vec, u_obs_vel)
                qolo_dir = np.arctan2(
                    agent.linear_velocity[1], agent.linear_velocity[0]
                )
                if qolo_dir != 0:
                    agent.pose.orientation = qolo_dir

            self.update_state_publisher(
                pose=agent.pose,
                frame_prefix=agent.name,
                object_type=agent.object_type,
            )

        self.publish_furniture_type(publish_type=ObjectType.CHAIR, base_name="chair")
        self.publish_furniture_type(publish_type=ObjectType.TABLE, base_name="table")
        self.publish_furniture_type(
            publish_type=ObjectType.HOSPITAL_BED, base_name="hospital_bed"
        )
        self.publish_furniture_type(publish_type=ObjectType.QOLO, base_name="qolo")

    def publish_furniture_type(self, publish_type: ObjectType, base_name: str):
        it_obj = 0
        for ii, agent in enumerate(self.agent):
            if agent.object_type != publish_type:
                continue

            name = base_name + str(it_obj)
            self.update_state_publisher(agent.pose, name, object_type=publish_type)
            it_obj += 1
            print(f"publish {publish_type}")

    def step_forwards(self):
        print("Step forward")
        self.ii -= 1

    def step_backwards(self):
        print("Step back")
        self.ii += 1

    def pause_toggle(self):
        self._pause != self._pause

        if self._pause:
            print("Pause")
        else:
            print("Continue")

    def check_keypress(self):
        if keyboard.is_pressed("space"):
            self.pause_toggle()

        elif (
            keyboard.is_pressed("right")
            or keyboard.is_pressed("j")
            or keyboard.is_pressed("n")
        ):
            self.step_forward()

        elif (
            keyboard.is_pressed("left")
            or keyboard.is_pressed("k")
            or keyboard.is_pressed("p")
        ):
            self.step_back()

        elif keyboard.is_pressed("esc") or keyboard.is_pressed("q"):
            print("Shutdown.")
            # self.shutdown()
            rclpy.shutdown()
