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

from autonomous_furniture.agent_helper_functions import (
    compute_ang_weights,
    compute_drag_angle,
    compute_goal_angle,
    compute_gamma_critic,
    apply_velocity_constraints,
    ctr_point_vel_from_agent_kinematics,
    compute_ctr_point_vel_from_obs_avoidance,
    agent_kinematics_from_ctr_point_vel,
    apply_velocity_constraints,
    apply_linear_and_angular_acceleration_constraints,
    evaluate_safety_repulsion
)

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
        shape_container: ObstacleContainer,
        shape_positions: Optional[np.ndarray],
        obstacle_environment: ObstacleContainer,
        priority_value: float = 1.0,
        control_points: Optional[np.ndarray] = None,
        parking_pose: ObjectPose = None,
        goal_pose: ObjectPose = None,
        name: str = "no_name",
        static: bool = False,
        object_type: ObjectType = ObjectType.OTHER,
        symmetry: Optional[float] = None,
        gamma_critic: float = 0.0,
        d_critic: float = 1.0,
        gamma_critic_max: float = 2.0,
        gamma_critic_min: float = 1.2,
        gamma_stop: float = 1.1,
    ) -> None:
        self._shape_container = shape_container
        self.object_type = object_type
        self.maximum_linear_velocity = 1.0  # m/s
        self.maximum_angular_velocity = 1.0  # rad/s
        self.maximum_linear_acceleration = 4.0  # m/s^2
        self.maximum_angular_acceleration = 10.0  # rad/s^2

        self.symmetry = symmetry

        # Default values for new variables
        self.danger = False
        self.color = np.array([176, 124, 124]) / 255.0

        self.priority = priority_value
        self.virtual_drag = max(self._shape.axes_length) / min(self._shape.axes_length)
        # TODO maybe append the shape directly in bos env,
        # and then do a destructor to remove it from the list
        self._obstacle_environment = obstacle_environment
        self._control_points = control_points
        self._parking_pose = parking_pose
        self._shape_positions = shape_positions
        self._goal_pose = goal_pose

        # Adding the current shape of the agent to the list of
        # obstacle_env so as to be visible to other agents
        for i in enumerate(self._shape_container):
            self._obstacle_environment.append(self._shape_container[i])

        self._static = static
        self.name = name

        self.converged: bool = False
        # Emergency Stop
        self.stop: bool = False

        # metrics
        self.direct_distance = LA.norm(goal_pose.position - self.position)
        self.total_distance = 0
        self.time_conv = 0
        self.time_conv_direct = 0
        self._proximity = 0
        # Tempory attribute only use for the qualitative example of the report.
        # To be deleted after
        self._list_prox = []

        ##  Emergency stop values ##
        self.gamma_critic = gamma_critic
        # distance from which gamma_critic starts shrinking
        self.d_critic = d_critic
        # value of gamma_critic before being closer than d_critic
        self.gamma_critic_max = gamma_critic_max
        # minimal value of gamma_critic as it should stay vigilant
        # and make space even at the goal
        self.gamma_critic_min = gamma_critic_min
        # agent should stop when a ctrpoint reaches a gamma value under this threshold
        self.gamma_stop = gamma_stop

        self.linea_velocity = 0.0
        self.angular_velocity = 0.0

    def update_velocity(self) -> None:
        self,
        mini_drag: str = ("nodrag",)
        version: str = ("v1",)
        emergency_stop: bool = (True,)
        safety_module: bool = (True,)
        time_step: float = (0.04,)

        # if static velocities will always be 0 per definition
        if self._static:
            self.linear_velocity = np.array([0.0, 0.0])
            self.angular_velocity = 0.0
            return

        # save the past commands to be able to check kinematic constraints
        linear_velocity_old = self.linear_velocity
        if self.angular_velocity == None:
            self.angular_velocity = 0.0
        angular_velocity_old = self.angular_velocity

        environment_without_me = self.get_obstacles_without_me()
        if not len(environment_without_me):
            self.linear_velocity = self._dynamics.evaluate(self.position)
            self.angular_velocity = 0

        global_control_points = self.get_global_control_points()

        weights = self.get_weight_of_control_points(
            global_control_points, environment_without_me
        )

        # plt.arrow(self.position[0], self.position[1], initial_velocity[0],
        #       initial_velocity[1], head_width=0.1, head_length=0.2, color='g')
        d = LA.norm(self.position - self._goal_pose.position)

        if version == "v2":
            initial_velocity = linear_velocity_old.copy()
            # plt.arrow(self.position[0], self.position[1], initial_velocity[0], initial_velocity[1], head_width=0.1, head_length=0.2, color='m')
            # compute goal orientation wheights
            w1, w2 = self.compute_ang_weights(mini_drag, d)
            drag_angle = self.compute_drag_angle(initial_velocity)
            goal_angle = self.compute_goal_angle()
            # TODO Very clunky : Rather make a function out of it
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computedenv
            desired_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)
            # plt.arrow(ctp[0], ctp[1], init_velocities[0, ii],
            #           init_velocities[1, ii], head_width=0.1, head_length=0.2, color='g')
            # plt.arrow(ctp[0], ctp[1], velocities[0, ii], velocities[1,
            #           ii], head_width=0.1, head_length=0.2, color='m')
            velocities = self.ctr_point_vel_from_agent_kinematics(
                desired_angular_vel, initial_velocity
            )

        elif version == "v1":
            velocities = self.compute_ctr_point_vel_from_obs_avoidance()

        ### CHECK WHETHER TO ADAPT THE AGENT'S KINEMATICS TO THE CURRENT OBSTACLE SITUATION ###
        if (
            safety_module or emergency_stop
        ):  # collect the gamma values of all the control points
            gamma_values = np.zeros(
                global_control_points.shape[1]
            )  # Store the min Gamma of each control point
            obs_idx = [None] * global_control_points.shape[
                1
            ]  # Idx of the obstacle in the environment where the Gamma is calculated from

            for ii in range(global_control_points.shape[1]):
                (
                    obs_idx[ii],
                    gamma_values[ii],
                ) = self.get_gamma_product_crowd(  # TODO: Done elsewhere, for efficiency maybe will need to be delete
                    global_control_points[:, ii], environment_without_me
                )

            if emergency_stop:
                # if any gamma values are lower od equal gamma_stop
                if any(x <= self.gamma_stop for x in gamma_values):
                    # print("EMERGENCY STOP")
                    self.angular_velocity = 0
                    self.linear_velocity = np.array([0.0, 0.0])
                    return

            if safety_module:
                self.compute_gamma_critic(d)
                # Check if the gamma function is below gramma_critic
                list_critic_gammas_indx = []
                for ii in range(global_control_points.shape[1]):
                    if gamma_values[ii] < self.gamma_critic:
                        list_critic_gammas_indx.append(ii)
                        self.color = "k"  # np.array([221, 16, 16]) / 255.0
                if len(list_critic_gammas_indx) > 0:
                    velocities = self.evaluate_safety_repulsion(
                        list_critic_gammas_indx=list_critic_gammas_indx,
                        environment_without_me=environment_without_me,
                        global_control_points=global_control_points,
                        obs_idx=obs_idx,
                        gamma_values=gamma_values,
                        velocities=velocities,
                    )

        self.agent_kinematics_from_ctr_point_vel(velocities, weights)
