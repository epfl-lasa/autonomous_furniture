from typing import Optional
import time

import math
from math import sin, cos, pi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import keyboard

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

from tf2_ros import TransformBroadcaster, TransformStamped
import tf_transformations

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
from autonomous_furniture.dynamical_system_animation3D import DynamicalSystemAnimation3D

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
from autonomous_furniture.agent_helper_functions import (
    update_multi_layer_simulation,
    plot_animation,
)

from autonomous_furniture.furniture_creators import (
    create_3D_table_surface_legs,
    create_3D_chair,
)

import pathlib
import yaml
import argparse
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray


def get_layers(parameter_file):
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

    return [layer_lower, layer_upper]


class ExperimentNode(Node):
    # it_max: int
    # dt_sleep: float = 0.05
    # dt_simulation: float = 0.05
    def __init__(
        self,
    ) -> None:
        # rclpy.init()
        super().__init__("experimet_node")

        parameter_file = (
            str(pathlib.Path(__file__).parent.resolve())
            + "/parameters/experiment_simulation.yaml"
        )

        self.dim = 2
        self.layer_list = get_layers(parameter_file)
        self.number_layer = len(self.layer_list)
        self.number_agent = len(self.layer_list[0])

        self.node_name = self.get_name()
        self.ii = 0

        self.furniture_poses = []
        for i in range(self.number_agent):
            for k in range(self.number_layer):
                if not self.layer_list[k][i] == None:
                    self.furniture_poses.append(
                        ObjectPose(
                            position=self.layer_list[k][i]._reference_pose.position,
                            orientation=self.layer_list[k][
                                i
                            ]._reference_pose.orientation,
                        )
                    )
                    break
        # add also time information
        self.furniture_poses.append(0.0)

        # clone furniture poses list to create one for past furniture poses to calculate velocity
        self.past_furniture_poses = self.furniture_poses.copy()

        self.time = 0.0

        self.mobile_robot_publisher = self.create_publisher(Twist, "cmd_vel", 50)

        self.vicon_publisher_chair1 = self.create_publisher(
            PoseStamped, "groundtruth_chair1", 50
        )
        self.vicon_publisher_chair2 = self.create_publisher(
            PoseStamped, "groundtruth_chair2", 50
        )
        self.vicon_publisher_mobile_table = self.create_publisher(
            PoseStamped, "groundtruth_mobile_table", 50
        )
        self.vicon_publisher_static_table = self.create_publisher(
            PoseStamped, "groundtruth_static_table", 50
        )

        self.vicon_subscriber_chair1 = self.create_subscription(
            PoseStamped, "groundtruth_chair1", self.vicon_callback, 50
        )
        self.vicon_subscriber_chair2 = self.create_subscription(
            PoseStamped, "groundtruth_chair2", self.vicon_callback, 50
        )
        self.vicon_subscriber_mobile_table = self.create_subscription(
            PoseStamped, "groundtruth_mobile_table", self.vicon_callback, 50
        )
        self.vicon_subscriber_static_table = self.create_subscription(
            PoseStamped, "groundtruth_static_table", self.vicon_callback, 50
        )

        self.agent_pos_saver = []
        for i in range(self.number_agent):
            self.agent_pos_saver.append([])
        for i in range(self.number_agent):
            saved = False
            for k in range(self.number_layer):
                if not self.layer_list[k] == None:
                    if (
                        not saved
                    ):  # make sure the positions are only saved once when an agent is present in multiple layers
                        self.agent_pos_saver[i].append(
                            self.layer_list[k][i]._reference_pose.position
                        )
                        saved = True

        with open(parameter_file, "r") as openfile:
            yaml_object = yaml.safe_load(openfile)
        self.x_lim = yaml_object["x limit"]
        self.y_lim = yaml_object["y limit"]
        figsize = yaml_object["figure size"]
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=120)
        self.obstacle_colors = yaml_object["obstacle colors"]
        
        
        self.timer_period = 0.02  # seconds
        self.timer = self.create_timer(self.timer_period, self.update_step)

        self.it=0

    def update_step(self) -> None:
        self.update_states_with_vicon_feedback()

        self.layer_list, self.agent_pos_saver = update_multi_layer_simulation(
            number_layer=self.number_layer,
            number_agent=self.number_agent,
            layer_list=self.layer_list,
            dt_simulation=self.timer_period,
            agent_pos_saver=self.agent_pos_saver,
        )

        plot_animation(
            self.ax,
            self.layer_list,
            self.number_layer,
            self.number_agent,
            self.obstacle_colors,
            self.agent_pos_saver,
            self.x_lim,
            self.y_lim,
        )
        plt.pause(0.02)

        self.publish_fake_vicon_groundtruth()

        self.send_mobile_robot_commands()

    def send_mobile_robot_commands(self):
        cmd_vel = Twist()

        for i in range(self.number_agent):
            for k in range(self.number_layer):
                if not self.layer_list[k][i] == None:
                    cmd_vel.linear.x = self.layer_list[k][i].linear_velocity[0]
                    cmd_vel.linear.y = self.layer_list[k][i].linear_velocity[1]

                    cmd_vel.angular.z = self.layer_list[k][i].angular_velocity
                    self.mobile_robot_publisher.publish(cmd_vel)
                    break

    def vicon_callback(self, vicon_msg):
        # copy old information into past furniture pose list
        self.past_furniture_poses = self.furniture_poses.copy()

        quaternion = [
            vicon_msg.pose.orientation.x,
            vicon_msg.pose.orientation.y,
            vicon_msg.pose.orientation.z,
            vicon_msg.pose.orientation.w,
        ]

        # copy new information into furniture pose list
        if vicon_msg.header.frame_id == "chair1":
            self.furniture_poses[0].position = np.array(
                [vicon_msg.pose.position.x, vicon_msg.pose.position.y]
            )
            self.furniture_poses[
                0
            ].orientation = tf_transformations.euler_from_quaternion(quaternion)[2]
        elif vicon_msg.header.frame_id == "chair2":
            self.furniture_poses[1].position = np.array(
                [vicon_msg.pose.position.x, vicon_msg.pose.position.y]
            )
            self.furniture_poses[
                1
            ].orientation = tf_transformations.euler_from_quaternion(quaternion)[2]
        elif vicon_msg.header.frame_id == "mobile_table":
            self.furniture_poses[2].position = np.array(
                [vicon_msg.pose.position.x, vicon_msg.pose.position.y]
            )
            self.furniture_poses[
                2
            ].orientation = tf_transformations.euler_from_quaternion(quaternion)[2]
        elif vicon_msg.header.frame_id == "static_table":
            self.furniture_poses[3].position = np.array(
                [vicon_msg.pose.position.x, vicon_msg.pose.position.y]
            )
            self.furniture_poses[
                3
            ].orientation = tf_transformations.euler_from_quaternion(quaternion)[2]
            self.furniture_poses[-1] = (
                vicon_msg.header.stamp.sec + vicon_msg.header.stamp.nanosec * 1e-9
            )

    def update_states_with_vicon_feedback(self):
        # dt = self.furniture_poses[-1] - self.past_furniture_poses[-1]

        # if dt == 0.0:
        #     breakpoint()

        for i in range(self.number_agent):
            for k in range(self.number_layer):
                if not self.layer_list[k][i] == None:
                    # update velocity
                    self.layer_list[k][i].linear_velocity = (
                        self.furniture_poses[i].position
                        - self.past_furniture_poses[i].position
                    ) / self.timer_period

                    delta_angle = (
                        self.furniture_poses[i].orientation
                        - self.past_furniture_poses[i].orientation
                    )
                    if delta_angle > np.pi:
                        delta_angle = 2 * np.pi - delta_angle
                    elif delta_angle < -np.pi:
                        delta_angle = -(delta_angle + 2 * np.pi)

                    self.layer_list[k][i].angular_velocity = (
                        delta_angle / self.timer_period
                    )  # fix for step from 0 to 2pi

                    # update reference pose
                    self.layer_list[k][i]._reference_pose = ObjectPose(
                        position=self.furniture_poses[i].position,
                        orientation=self.furniture_poses[i].orientation,
                    )
                    # update shape poses
                    for s in range(len(self.layer_list[k][i]._shape_list)):
                        self.layer_list[k][i]._shape_list[s].position = self.layer_list[
                            k
                        ][i]._reference_pose.transform_position_from_relative(
                            np.copy(self.layer_list[k][i]._shape_positions[s])
                        )
                        self.layer_list[k][i]._shape_list[
                            s
                        ].orientation = self.layer_list[k][
                            i
                        ]._reference_pose.orientation

    def publish_fake_vicon_groundtruth(self):
        chair1_pose = PoseStamped()
        chair1_pose.header.frame_id = "chair1"
        chari2_pose = PoseStamped()
        chari2_pose.header.frame_id = "chair2"
        mobile_table_pose = PoseStamped()
        mobile_table_pose.header.frame_id = "mobile_table"
        static_table_pose = PoseStamped()
        static_table_pose.header.frame_id = "static_table"

        pose_list = [chair1_pose, chari2_pose, mobile_table_pose, static_table_pose]
        publisher_list = [
            self.vicon_publisher_chair1,
            self.vicon_publisher_chair2,
            self.vicon_publisher_mobile_table,
            self.vicon_publisher_static_table,
        ]

        for i in range(self.number_agent):
            for k in range(self.number_layer):
                if not self.layer_list[k][i] == None:
                    pose_list[i].pose.position.x = self.layer_list[k][
                        i
                    ]._reference_pose.position[0]
                    pose_list[i].pose.position.y = self.layer_list[k][
                        i
                    ]._reference_pose.position[1]

                    quaternion = tf_transformations.quaternion_from_euler(
                        0.0, 0.0, self.layer_list[k][i]._reference_pose.orientation
                    )
                    pose_list[i].pose.orientation.x = quaternion[0]
                    pose_list[i].pose.orientation.y = quaternion[1]
                    pose_list[i].pose.orientation.z = quaternion[2]
                    pose_list[i].pose.orientation.w = quaternion[3]

                    pose_list[i].header.stamp = self.get_clock().now().to_msg()

                    publisher_list[i].publish(pose_list[i])
                    break


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    rclpy.init(args=None)

    experiment_node = ExperimentNode()
    experiment_node.publish_fake_vicon_groundtruth()

    rclpy.spin(experiment_node)
