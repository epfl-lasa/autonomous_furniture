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
from autonomous_furniture.agent3D import Furniture3D

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
from autonomous_furniture.agent_helper_functions import update_multi_layer_simulation

from geometry_msgs.msg import Twist

class ExperimentNode(Node):
    # it_max: int
    # dt_sleep: float = 0.05
    # dt_simulation: float = 0.05
    def __init__(
        self,
        layer_list: list[Furniture3D],
    ) -> None:

        # rclpy.init()
        super().__init__("experimet_node")

        self.dim = 2
        self.number_layer = len(layer_list)
        self.number_agent = len(layer_list[0])
        self.layer_list = layer_list

        self.node_name = self.get_name()
        self.ii = 0
        
        self.mobile_robot_publisher = self.create_publisher(Twist, "cmd_vel", 50)

    def update_step(self) -> None:

        self.update_states_with_vicon_feedback()

        self.layer_list = update_multi_layer_simulation(
            number_layer=self.number_layer,
            number_agent=self.number_agent,
            layer_list=self.layer_list,
        )

        self.send_mobile_robot_commands()
        
    def send_mobile_robot_commands(self):
        cmd_vel = Twist()
        
        for i in range(self.number_agent):
            for k in range(self.number_layer):
                if not self.layer_list[k][i]==None:
                    cmd_vel.linear=self.layer_list[k][i].linear_velocity
                    cmd_vel.angular=self.layer_list[k][i].angular_velocity
                    break
        
        self.mobile_robot_publisher.publish(cmd_vel)