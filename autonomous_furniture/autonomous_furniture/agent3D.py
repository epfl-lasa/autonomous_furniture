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
    get_weight_of_control_points,
    get_params_from_file,
)


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
        shape_positions: Optional[np.ndarray],
        obstacle_environment: ObstacleContainer,
        control_points: Optional[np.ndarray],
        starting_pose: ObjectPose,
        goal_pose: ObjectPose,
        parameter_file: str,
        priority_value: float = None,
        parking_pose: ObjectPose = None,
        name: str = "no_name",
        static: bool = None,
        object_type: ObjectType = ObjectType.OTHER,
        min_drag: bool = None,
        soft_decoupling: bool = None,
        safety_module: bool = None,
        emergency_stop: bool = None,
        d_critic: float = None,
        gamma_critic_max: float = None,
        gamma_critic_min: float = None,
        gamma_stop: float = None,
        safety_damping: float = None,
        cutoff_gamma_weights: float = None,
        cutoff_gamma_obs: float = None,
        maximum_linear_velocity: float = None,  # m/s
        maximum_angular_velocity: float = None,  # rad/s
        maximum_linear_acceleration: float = None,  # m/s^2
        maximum_angular_acceleration: float = None,  # rad/s^2
    ) -> None:
        # safe mandatory variables
        self._shape_list = shape_list
        self._goal_pose = goal_pose
        self._reference_pose = starting_pose
        self._parking_pose = parking_pose
        self._shape_positions = shape_positions
        self._control_points = control_points
        self._obstacle_environment = obstacle_environment

        self = get_params_from_file(
            self,
            parameter_file,
            min_drag,
            soft_decoupling,
            safety_module,
            emergency_stop,
            maximum_linear_velocity,
            maximum_angular_velocity,
            maximum_linear_acceleration,
            maximum_angular_acceleration,
            safety_damping,
            gamma_critic_max,
            gamma_critic_min,
            gamma_stop,
            d_critic,
            cutoff_gamma_weights,
            cutoff_gamma_obs,
            static,
            name,
            priority_value,
        )

        for i in range(len(shape_list)):
            self._shape_list[i].reactivity = self.priority

        # Adding the current shape of the agent to the list of
        # obstacle_env so as to be visible to other agents
        for i in range(len(self._shape_list)):
            self._obstacle_environment.append(self._shape_list[i])

        self.converged: bool = False
        # Emergency Stop
        self.stop: bool = False

        # metrics
        self.direct_distance = LA.norm(
            goal_pose.position - self._reference_pose.position
        )
        self.total_distance = 0
        self.time_conv = 0
        self.time_conv_direct = 0
        self._proximity = 0
        # Tempory attribute only use for the qualitative example of the report.
        # To be deleted after
        self._list_prox = []

        ##  Emergency stop values ##
        self.gamma_critic = 0.0

        self.linear_velocity = np.array([0.0, 0.0])
        self.angular_velocity = 0.0

        self.min_gamma = 1e6
        self.ctr_pt_number = self._control_points.shape[0]
        self.object_type = object_type

    def get_obstacles_without_me(self):
        obs_env_without_me = []
        for i in range(len(self._obstacle_environment)):
            obs = self._obstacle_environment[i]
            save_obs = True
            for j in range(len(self._shape_list)):
                shape = self._shape_list[j]
                if obs == shape:
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
        # set the shape's linear and angular velocity, maybe not the right place do define it once we try multiple layers?
        for i in range(len(self._shape_list)):
            if self.static:
                self._shape_list[i].linear_velocity = 0.0
                self._shape_list[i].angular_velocity = np.zeros(2)
            else:
                # angular velocity in rigid bodies is always the same in each point
                self._shape_list[i].angular_velocity = self.angular_velocity
                # linear velocity follows the gemeral rigid body equation
                shape_position_global = (
                    self._reference_pose.transform_position_from_relative(
                        self._shape_positions[i].copy()
                    )
                )
                shape_position_global_wrt_ref = (
                    shape_position_global - self._reference_pose.position
                )
                self._shape_list[i].linear_velocity[0] = (
                    self.linear_velocity[0]
                    - self.angular_velocity * shape_position_global_wrt_ref[1]
                )
                self._shape_list[i].linear_velocity[1] = (
                    self.linear_velocity[1]
                    + self.angular_velocity * shape_position_global_wrt_ref[0]
                )

    def do_velocity_step(self, dt):
        self.update_shape_kinematics()
        for i in range(len(self._shape_list)):
            self._shape_list[i].do_velocity_step(dt)
        self._reference_pose.update(
            dt, ObjectTwist(linear=self.linear_velocity, angular=self.angular_velocity)
        )

    def apply_kinematic_constraints(self):
        linear_velocity = np.copy(self.linear_velocity)
        angular_velocity = self.angular_velocity

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
            self.linear_velocity_old,
            self.angular_velocity_old,
            linear_velocity,
            angular_velocity,
            maximum_linear_acceleration=self.maximum_linear_acceleration,
            maximum_angular_acceleration=self.maximum_angular_acceleration,
            time_step=self.time_step,
        )
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

    def update_velocity(
        self,
        time_step: float = 0.1,
    ) -> None:
        # if static velocities will always be 0 per definition
        if self.static:
            self.linear_velocity = np.array([0.0, 0.0])
            self.angular_velocity = 0.0
            self.stop = True
            return
        self.time_step = time_step
        # save the past commands to be able to check kinematic constraints
        self.linear_velocity_old = self.linear_velocity
        if self.angular_velocity == None:
            self.angular_velocity = 0.0
        self.angular_velocity_old = self.angular_velocity

        global_control_points = self.get_global_control_points()
        goal_control_points = self.get_goal_control_points()
        environment_without_me = self.get_obstacles_without_me()

        if bool(environment_without_me):  # if there are other objects to take care of
            weights = get_weight_of_control_points(
                global_control_points,
                environment_without_me,
                cutoff_gamma=self.cutoff_gamma_weights,
                gamma0=self.gamma0,
                frac_gamma_nth=self.frac_gamma_nth,
            )
        else:
            weights = np.ones(self.ctr_pt_number) / self.ctr_pt_number

        d = LA.norm(self._reference_pose.position - self._goal_pose.position)

        if self.soft_decoupling:
            if LA.norm(self.linear_velocity_old) < 1e-6:
                initial_velocity = (
                    self._goal_pose.position - self._reference_pose.position
                )
                initial_velocity = (
                    initial_velocity
                    / LA.norm(initial_velocity)
                    * self.maximum_linear_velocity
                )
            else:
                initial_velocity = self.linear_velocity_old.copy()

            # compute goal orientation wheights
            w1, w2 = compute_ang_weights(self.min_drag, d, self.virtual_drag, self.k, self.alpha)
            drag_angle = compute_drag_angle(
                initial_velocity, self._reference_pose.orientation
            )
            goal_angle = (
                compute_goal_angle(  ##compute orientation difference to reach goal
                    self._goal_pose.orientation, self._reference_pose.orientation
                )
            )
            # TODO Very clunky : Rather make a function out of it
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computedenv
            initial_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)

            velocities_from_DSM = compute_ctr_point_vel_from_obs_avoidance(
                number_ctrpt=self.ctr_pt_number,
                goal_pos_ctr_pts=np.copy(goal_control_points),
                actual_pos_ctr_pts=np.copy(global_control_points),
                environment_without_me=self.get_obstacles_without_me(),
                priority=self.priority,
                cutoff_gamma_obs=self.cutoff_gamma_obs,
            )

            linear_velocity, angular_velocity = agent_kinematics_from_ctr_point_vel(
                velocities_from_DSM,
                weights,
                global_control_points=np.copy(global_control_points),
                ctrpt_number=self.ctr_pt_number,
                global_reference_position=self._reference_pose.position,
            )

            velocities = ctr_point_vel_from_agent_kinematics(
                initial_angular_vel,
                linear_velocity,
                number_ctrpt=self.ctr_pt_number,
                global_control_points=np.copy(global_control_points),
                environment_without_me=environment_without_me,
                priority=self.priority,
                DSM=True,
                reference_position=self._reference_pose.position,
                cutoff_gamma_obs=self.cutoff_gamma_obs,
            )

        else:
            velocities = compute_ctr_point_vel_from_obs_avoidance(
                number_ctrpt=self.ctr_pt_number,
                goal_pos_ctr_pts=np.copy(goal_control_points),
                actual_pos_ctr_pts=np.copy(global_control_points),
                environment_without_me=environment_without_me,
                priority=self.priority,
                cutoff_gamma_obs=self.cutoff_gamma_obs,
            )

        if not bool(
            environment_without_me
        ):  # in case there are no significant obstacle no safety modulations are needed
            (
                linear_velocity,
                angular_velocity,
            ) = agent_kinematics_from_ctr_point_vel(
                velocities,
                weights,
                global_control_points=np.copy(global_control_points),
                ctrpt_number=self.ctr_pt_number,
                global_reference_position=self._reference_pose.position,
            )
            self.linear_velocity = linear_velocity
            self.angular_velocity = angular_velocity
            return

        ### gett gamma values and save the smallest one
        gamma_values = np.zeros(
            self.ctr_pt_number
        )  # Store the min Gamma of each control point
        obs_idx = [None] * global_control_points.shape[
            1
        ]  # Idx of the obstacle in the environment where the Gamma is calculated from

        for ii in range(self.ctr_pt_number):
            (
                obs_idx[ii],
                gamma_values[ii],
            ) = get_gamma_product_crowd(
                global_control_points[:, ii], environment_without_me
            )
        # print("gamma_values:\n", gamma_values)
        self.min_gamma = np.amin(gamma_values)

        ### CHECK WHETHER TO ADAPT THE AGENT'S KINEMATICS TO THE CURRENT OBSTACLE SITUATION ###
        if (
            self.safety_module or self.emergency_stop
        ):  # collect the gamma values of all the control points
            if self.emergency_stop:
                # if any gamma values are lower od equal gamma_stop
                if any(x <= self.gamma_stop for x in gamma_values):
                    # print("EMERGENCY STOP")
                    self.angular_velocity = 0
                    self.linear_velocity = np.array([0.0, 0.0])
                    self.stop = True
                    self.stopped = True
                    return
                else:
                    self.stop = False

            if self.safety_module:
                self.gamma_critic = compute_gamma_critic(
                    d=d,
                    d_critic=self.d_critic,
                    gamma_critic_max=self.gamma_critic_max,
                    gamma_critic_min=self.gamma_critic_min,
                )
                # Check if the gamma function is below gramma_critic
                list_critic_gammas_indx = []
                for ii in range(self.ctr_pt_number):
                    if gamma_values[ii] < self.gamma_critic:
                        list_critic_gammas_indx.append(ii)
                if len(list_critic_gammas_indx) > 0:
                    # calculate the real velocties of the control points after weighting before applying safety module
                    (
                        linear_velocity,
                        angular_velocity,
                    ) = agent_kinematics_from_ctr_point_vel(
                        velocities,
                        weights,
                        global_control_points=np.copy(global_control_points),
                        ctrpt_number=self.ctr_pt_number,
                        global_reference_position=self._reference_pose.position,
                    )

                    self.linear_velocity = linear_velocity
                    self.angular_velocity = angular_velocity
                    self.apply_kinematic_constraints()
                    linear_velocity = np.copy(self.linear_velocity)
                    angular_velocity = self.angular_velocity

                    velocities = ctr_point_vel_from_agent_kinematics(
                        angular_velocity,
                        linear_velocity,
                        number_ctrpt=self.ctr_pt_number,
                        global_control_points=np.copy(global_control_points),
                        environment_without_me=environment_without_me,
                        priority=self.priority,
                        DSM=False,
                        reference_position=self._reference_pose.position,
                        cutoff_gamma_obs=self.cutoff_gamma_obs,
                    )
                    velocities = evaluate_safety_repulsion(
                        list_critic_gammas_indx=list_critic_gammas_indx,
                        environment_without_me=environment_without_me,
                        global_control_points=np.copy(global_control_points),
                        obs_idx=obs_idx,
                        gamma_values=gamma_values,
                        velocities=velocities,
                        gamma_critic=self.gamma_critic,
                        local_control_points=self._control_points,
                        safety_damping=self.safety_damping,
                    )

        linear_velocity, angular_velocity = agent_kinematics_from_ctr_point_vel(
            velocities,
            weights,
            global_control_points=np.copy(global_control_points),
            ctrpt_number=self.ctr_pt_number,
            global_reference_position=self._reference_pose.position,
        )

        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
