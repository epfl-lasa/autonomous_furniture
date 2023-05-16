from typing import Optional
import time

import math
from math import sin, cos, pi

import numpy as np
import matplotlib.pyplot as plt

import keyboard

import rospy
import tf

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

from autonomous_furniture.furniture_creators import (
    create_3D_table_surface_legs,
    create_3D_chair,
)

import pathlib


from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray

def get_layers():
    parameter_file = (
        str(pathlib.Path(__file__).parent.resolve())
        + "/parameters/experiment_simulation.json"
    )

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


class ExperimentNode():
    # it_max: int
    # dt_sleep: float = 0.05
    # dt_simulation: float = 0.05
    def __init__(
        self,
    ) -> None:

        # rclpy.init()
        super().__init__("experimet_node")

        self.dim = 2
        self.layer_list = get_layers()
        self.number_layer = len(self.layer_list)
        self.number_agent = len(self.layer_list[0])

        self.node_name = self.get_name()
        self.ii = 0
        
        self.furniture_poses = []
        for i in range(self.number_agent):
            for k in range(self.number_layer):
                if not self.layer_list[k][i]==None:
                    self.furniture_poses.append(ObjectPose(position=self.layer_list[k][i].reference_pose.position, orientation=self.layer_list[k][i].reference_pose.orientation))
                    break
        #add also time information
        self.furniture_poses.append(0.0)
        
        #clone furniture poses list to create one for past furniture poses to calculate velocity
        self.past_furniture_poses = self.furniture_poses.copy()
        
        self.time = 0.0
        
        self.mobile_robot_publisher = rospy.Publisher("cmd_vel",Twist, 50)
        self.vicon_subscriber=rospy.Subscriber("/groundtruth", MarkerArray, self.vicon_callback)
        
    def update_step(self) -> None:
        
        self.update_states_with_vicon_feedback()

        self.layer_list = update_multi_layer_simulation(
            number_layer=self.number_layer,
            number_agent=self.number_agent,
            layer_list=self.layer_list,
            dt_simulation=0.02
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
        
    def vicon_callback(self, vicon_msg):
        #copy old information into past furniture pose list
        self.past_furniture_poses = self.furniture_poses.copy()

        #copy new information into furniture pose list
        for i in range(len(vicon_msg.markers)):
            self.furniture_poses[i].position = np.array([vicon_msg.markers[i].pose.position.x, vicon_msg.markers[i].pose.position.y, vicon_msg.markers[i].pose.position.z])
            self.furniture_poses[i].orientation = tf.transformations.euler_from_quaternion(vicon_msg.markers[i].pose.orientation)[2]

        self.furniture_poses[-1] = vicon_msg.markers[0].header.stamp.secs + vicon_msg.markers[0].header.stamp.nsecs*1e-9
    
    def update_states_with_vicon_feedback(self):
        
        dt = self.furniture_poses[-1] - self.past_furniture_poses[-1]
        
        for i in range(self.number_agent):
            for k in range(self.number_layer):
                if not self.layer_list[k][i]==None:
                    #update velocity 
                    self.layer_list[k][i].linear_velocity = (self.furniture_poses[i].position - self.past_furniture_poses[i].position)/dt
                    self.layer_list[k][i].angular_velocity = (self.furniture_poses[i].orientation - self.past_furniture_poses[i].orientation)/dt

                    #update reference pose
                    self.layer_list[k][i].reference_pose = ObjectPose(position=self.furniture_poses[i].position, orientation=self.furniture_poses[i].orientation)
                    #update shape poses
                    for s in self.layer_list[k][i].shape_list
                    