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
    evaluate_safety_repulsion,
    get_gamma_product_crowd,
    get_weight_of_control_points
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
        shape_list: list[Obstacle],
        obstacle_environment: ObstacleContainer,
        control_points: Optional[np.ndarray],
        shape_positions: Optional[np.ndarray] = None,
        priority_value: float = 1.0,
        starting_pose: ObjectPose = None,
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
        self._shape_list = shape_list
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
        for i in range(len(shape_list)):
            self._shape_list[i].reactivity=self.priority
            
        if len(shape_list) == 1:
            self.virtual_drag = max(self._shape_list[0].axes_length) / min(self._shape_list[0].axes_length)
        else:
            self.virtual_drag = 1
        # TODO maybe append the shape directly in bos env,
        # and then do a destructor to remove it from the list
        self._obstacle_environment = obstacle_environment
        self._control_points = control_points
        self._parking_pose = parking_pose
        self._goal_pose = goal_pose
        
        if starting_pose==None:
            if len(shape_list) == 1:
                self._reference_pose = ObjectPose(position=shape_list[0].pose.position, orientation=shape_list[0].pose.orientation)
                self._shape_positions = np.array([[0.0, 0.0]])
            else:
                raise Exception("Please define a starting pose if agent has more than one shape!") 
        else:
            self._reference_pose = starting_pose
            self._shape_positions = shape_positions

        # Adding the current shape of the agent to the list of
        # obstacle_env so as to be visible to other agents
        for i in range(len(self._shape_list)):
            self._obstacle_environment.append(self._shape_list[i])

        self._static = static
        self.name = name

        self.converged: bool = False
        # Emergency Stop
        self.stop: bool = False

        # metrics
        self.direct_distance = LA.norm(goal_pose.position - self._reference_pose.position)
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

        self.linear_velocity = np.array([0.0, 0.0])
        self.angular_velocity = 0.0

    def get_obstacles_without_me(self):
        obs_env_without_me = []
        for i in range(len(self._obstacle_environment)):
            obs = self._obstacle_environment[i]
            save_obs=True
            for j in range(len(self._shape_list)):
                shape=self._shape_list[j]
                if obs==shape:
                    save_obs = False
            if save_obs:
                obs_env_without_me.append(obs)
        return obs_env_without_me

    def get_global_control_points(self):
        return np.array(
            [
                self._reference_pose.transform_position_from_relative(ctp)
                for ctp in self._control_points
            ]
        ).T

    def get_goal_control_points(self):
        """Get goal-control-points in global frame."""
        return np.array(
            [
                self._goal_pose.transform_position_from_relative(ctp)
                for ctp in self._control_points
            ]
        ).T

    def update_shape_kinematics(self):
        #set the shape's linear and angular velocity, maybe not the right place do define it once we try multiple layers?
        for i in range(len(self._shape_list)):
            if self._static:
                self._shape_list[i].twist.angular=0.0
                self._shape_list[i].twist.linear=np.zeros(2)
            else:
                #angular velocity in rigid bodies is always the same in each point
                self._shape_list[i].twist.angular = self.angular_velocity
                #linear velocity follows the gemeral rigid body equation
                shape_position_global = self._reference_pose.transform_position_from_relative(self._shape_positions[i].copy())
                shape_position_global_wrt_ref = shape_position_global-self._reference_pose.position
                self._shape_list[i].twist.linear[0] = self.linear_velocity[0]-self.angular_velocity*shape_position_global_wrt_ref[1]
                self._shape_list[i].twist.linear[1] = self.linear_velocity[1]+self.angular_velocity*shape_position_global_wrt_ref[0]

    def do_velocity_step(self, dt):
        self.update_shape_kinematics()
        for i in range(len(self._shape_list)):
            self._shape_list[i].do_velocity_step(dt)
        self._reference_pose.update(dt, ObjectTwist(linear=self.linear_velocity, angular=self.angular_velocity))
        

    def update_velocity(
        self,
        mini_drag: str = "nodrag",
        version: str = "v1",
        emergency_stop: bool = True,
        safety_module: bool = True,
        time_step: float = 0.1,
    ) -> None:

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
        
        global_control_points = self.get_global_control_points()

        if not len(environment_without_me):
                raise Exception("NO OBSTACLES FOUND!")

        weights = get_weight_of_control_points(
            global_control_points, environment_without_me
        )

        # plt.arrow(self.position[0], self.position[1], initial_velocity[0],
        #       initial_velocity[1], head_width=0.1, head_length=0.2, color='g')
        d = LA.norm(self._reference_pose.position - self._goal_pose.position)

        if version == "v2":
            if LA.norm(linear_velocity_old)<1e-6:
                initial_velocity = self._goal_pose.position - self._reference_pose.position
                initial_velocity = initial_velocity/LA.norm(initial_velocity)*self.maximum_linear_velocity
            else:
                initial_velocity = linear_velocity_old.copy()
            # plt.arrow(self.position[0], self.position[1], initial_velocity[0], initial_velocity[1], head_width=0.1, head_length=0.2, color='m')
            # compute goal orientation wheights
            w1, w2 = compute_ang_weights(mini_drag, d, self.virtual_drag)
            drag_angle = compute_drag_angle(initial_velocity, self._reference_pose.orientation)
            goal_angle = compute_goal_angle(
                self._goal_pose.orientation, self._reference_pose.orientation
            )
            # TODO Very clunky : Rather make a function out of it
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computedenv
            initial_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)
            # plt.arrow(ctp[0], ctp[1], init_velocities[0, ii],
            #           init_velocities[1, ii], head_width=0.1, head_length=0.2, color='g')
            # plt.arrow(ctp[0], ctp[1], velocities[0, ii], velocities[1,
            #           ii], head_width=0.1, head_length=0.2, color='m')
            velocities_from_DSM = compute_ctr_point_vel_from_obs_avoidance(
                number_ctrpt=self._control_points.shape[0],
                goal_pos_ctr_pts=self.get_goal_control_points(),
                actual_pos_ctr_pts=self.get_global_control_points(),
                environment_without_me=self.get_obstacles_without_me(),
                priority=self.priority,
            )
            
            linear_velocity, angular_velocity = agent_kinematics_from_ctr_point_vel(
            velocities_from_DSM,
            weights,
            global_control_points=self.get_global_control_points(),
            ctrpt_number=self._control_points.shape[0],
            global_reference_position=self._reference_pose.position,
            )

            velocities = ctr_point_vel_from_agent_kinematics(
                initial_angular_vel,
                linear_velocity,
                number_ctrpt=self._control_points.shape[0],
                local_control_points=self._control_points,
                global_control_points=self.get_global_control_points(),
                actual_orientation=self._reference_pose.orientation,
                environment_without_me=self.get_obstacles_without_me(),
                priority=self.priority,
                DSM=True
            )

        elif version == "v1":
            velocities = compute_ctr_point_vel_from_obs_avoidance(
                number_ctrpt=self._control_points.shape[0],
                goal_pos_ctr_pts=self.get_goal_control_points(),
                actual_pos_ctr_pts=self.get_global_control_points(),
                environment_without_me=self.get_obstacles_without_me(),
                priority=self.priority,
            )
        ctp = self.get_global_control_points()
        for i in range(self._control_points.shape[0]):
            plt.arrow(ctp[0,i], ctp[1,i], velocities[0, i],
                    velocities[1, i], head_width=0.1, head_length=0.2, color='g')
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
                ) = get_gamma_product_crowd(
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
                self.gamma_critic = compute_gamma_critic(
                    d=d,
                    d_critic=self.d_critic,
                    gamma_critic_max=self.gamma_critic_max,
                    gamma_critic_min=self.gamma_critic_min,
                )
                # Check if the gamma function is below gramma_critic
                list_critic_gammas_indx = []
                for ii in range(global_control_points.shape[1]):
                    if gamma_values[ii] < self.gamma_critic:
                        list_critic_gammas_indx.append(ii)
                        self.color = "k"  # np.array([221, 16, 16]) / 255.0
                if len(list_critic_gammas_indx) > 0:
                    #calculate the real velocties of the control points after weighting before applying safety module
                    linear_velocity, angular_velocity = agent_kinematics_from_ctr_point_vel(
                        velocities,
                        weights,
                        global_control_points=self.get_global_control_points(),
                        ctrpt_number=self._control_points.shape[0],
                        global_reference_position=self._reference_pose.position,
                    )
                    velocities = ctr_point_vel_from_agent_kinematics(
                        angular_velocity,
                        linear_velocity,
                        number_ctrpt=self._control_points.shape[0],
                        local_control_points=self._control_points,
                        global_control_points=self.get_global_control_points(),
                        actual_orientation=self._reference_pose.orientation,
                        environment_without_me=self.get_obstacles_without_me(),
                        priority=self.priority,
                        DSM=False
                    )
                    velocities = evaluate_safety_repulsion(
                        list_critic_gammas_indx=list_critic_gammas_indx,
                        environment_without_me=environment_without_me,
                        global_control_points=global_control_points,
                        obs_idx=obs_idx,
                        gamma_values=gamma_values,
                        velocities=velocities,
                        gamma_critic=self.gamma_critic,
                        local_control_points = self._control_points,
                    )

        linear_velocity, angular_velocity = agent_kinematics_from_ctr_point_vel(
            velocities,
            weights,
            global_control_points=self.get_global_control_points(),
            ctrpt_number=self._control_points.shape[0],
            global_reference_position=self._reference_pose.position,
        )

        linear_velocity, angular_velocity = apply_velocity_constraints(
            linear_velocity,
            angular_velocity,
            maximum_linear_velocity=self.maximum_linear_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
        )

        (
            linear_velocity,
            angular_velocity,
        ) = apply_linear_and_angular_acceleration_constraints(
            linear_velocity_old,
            angular_velocity_old,
            linear_velocity,
            angular_velocity,
            maximum_linear_acceleration=self.maximum_linear_acceleration,
            maximum_angular_acceleration=self.maximum_angular_acceleration,
            time_step=time_step,
        )

        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity