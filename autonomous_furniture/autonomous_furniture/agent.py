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

# from vartools.states


class ObjectType(Enum):
    TABLE = auto()
    QOLO = auto()
    CHAIR = auto()
    HOSPITAL_BED = auto()
    OTHER = auto()


# def get_distance_to_obtacle_surface(
#     obstacle: Obstacle,
#     position: np.ndarray,
#     in_obstacle_frame: bool = True,
#     margin_absolut: Optional[float] = None,
#     in_global_frame: Optional[bool] = None,
# ) -> float:
#     if in_global_frame is not None:
#         in_obstacle_frame = not (in_global_frame)

#     if not in_obstacle_frame:
#         position = obstacle.pose.transform_position_from_reference_to_local(position)

#     if margin_absolut is None:
#         surface_point = obstacle.get_point_on_surface(
#             position=position,
#             in_obstacle_frame=True,
#         )
#     else:
#         surface_point = obstacle.get_point_on_surface(
#             position=position,
#             in_obstacle_frame=True,
#             margin_absolut=margin_absolut,
#         )

#     distance_surface = LA.norm(surface_point)
#     distance_position = LA.norm(position)

#     if distance_position > distance_surface:
#         distance = LA.norm(position - surface_point)
#     else:
#         distance = distance_position / distance_surface - 1

#     return distance


class BaseAgent(ABC):
    # Static variable, to trak the number of collisions within a scenario
    number_collisions = 0
    number_serious_collisions = 0

    def __init__(
        self,
        shape: Obstacle,
        obstacle_environment: ObstacleContainer,
        control_points: Optional[np.ndarray],
        goal_pose: ObjectPose,
        parameter_file: str,
        priority_value: float = None,
        parking_pose: ObjectPose = None,
        name: str = "no_name",
        static: bool = None,
        object_type: ObjectType = ObjectType.OTHER,
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
        super().__init__()

        self._shape = shape
        self.object_type = object_type
        self.virtual_drag = max(self._shape.axes_length) / min(self._shape.axes_length)
        self._obstacle_environment = obstacle_environment
        self._control_points = control_points
        self._parking_pose = parking_pose
        self._goal_pose = goal_pose
        # Adding the current shape of the agent to the list of
        # obstacle_env so as to be visible to other agents
        self._obstacle_environment.append(self._shape)

        self = get_params_from_file(
            self,
            parameter_file,
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

        self.converged: bool = False

        self.stop: bool = False
        self.ctr_pt_number = self._control_points.shape[0]

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
        self.gamma_critic = 0.0

    @property
    def pose(self):
        """Returns numpy-array position."""
        return self._shape.pose

    @property
    def position(self):
        """Returns numpy-array position."""
        return self._shape.pose.position

    @property
    def orientation(self) -> float:
        """Returns a (float) orientation (since uniquely 2d)"""
        if self._shape.pose.orientation is None:
            return 0
        else:
            return self._shape.pose.orientation

    @property
    def dimension(self):
        return self._shape.pose.dimension

    @property
    def linear_velocity(self):
        return self._shape.twist.linear

    @linear_velocity.setter
    def linear_velocity(self, value):
        self._shape.twist.linear = value

    @property
    def angular_velocity(self):
        return self._shape.twist.angular

    @angular_velocity.setter
    def angular_velocity(self, value):
        self._shape.twist.angular = value

    @property
    def priority(self):
        return self._shape.reactivity

    @priority.setter
    def priority(self, value):
        self._shape.reactivity = value

    @property
    def gamma_critic(self):
        return self._gamma_critic
        # return self._shape.gamma_critic

    @gamma_critic.setter
    def gamma_critic(self, value):
        # self._shape.gamma_critic = value
        self._gamma_critic = value

    @property
    def name(self):
        return self._shape.name

    @name.setter
    def name(self, name):
        self._shape.name = name

    # def get_obstacle_shape(self) -> Obstacle:
    #     return get_obstacle_shape()

    def do_velocity_step(self, dt):
        return self._shape.do_velocity_step(dt)

    def get_global_control_points(self):
        return np.array(
            [
                self._shape.pose.transform_position_from_relative(ctp)
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

    def get_veloctity_in_global_frame(self, velocity: npt.ArrayLike) -> np.ndarray:
        """Returns the transform of the velocity from relative to global frame."""
        return self._shape.pose.transform_direction_from_relative(np.array(velocity))

    def get_velocity_in_local_frame(self, velocity):
        """Returns the transform of the velocity from global to relative frame."""
        return self._shape.pose.transform_direction_to_relative(velocity)

    def get_obstacles_without_me(self):
        return [obs for obs in self._obstacle_environment if not obs == self._shape]


class Furniture(BaseAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dynamics = LinearSystem(
            attractor_position=self._goal_pose.position,
            maximum_velocity=self.maximum_linear_velocity,
        )

        # Seems to be used nowhere, mb to be removed
        self.minimize_drag: bool = False

        # Metrics
        self.time_conv_direct = self.direct_distance / self._dynamics.maximum_velocity

    @property
    def margin_absolut(self):
        return self._shape._margin_absolut

    def set_goal_pose(self, pose: ObjectPose) -> None:
        self._goal_pose = pose
        self.direct_distance = LA.norm(self._goal_pose.position - self.position)
        self._dynamics = LinearSystem(
            attractor_position=self._goal_pose.position,
            maximum_velocity=self.maximum_linear_velocity,
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
        mini_drag: str = "nodrag",
        version: str = "v1",
        emergency_stop: bool = True,
        safety_module: bool = True,
        time_step: float = 0.1,
    ) -> None:
        if self.static:
            self.linear_velocity = np.array([0.0, 0.0])
            self.angular_velocity = 0.0
            return

        self.time_step = time_step

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

        ### Calculate initial linear and angular velocity of the agent
        initial_velocity = self._dynamics.evaluate(self.position)

        d = LA.norm(self.position - self._goal_pose.position)

        if version == "v2":
            initial_velocity = obs_avoidance_interpolation_moving(
                position=self.position,
                initial_velocity=initial_velocity,
                obs=environment_without_me,
                self_priority=self.priority,
            )
            # compute goal orientation wheights
            w1, w2 = compute_ang_weights(
                mini_drag, d, self.virtual_drag, self.k, self.alpha
            )
            drag_angle = compute_drag_angle(initial_velocity, self.orientation)
            goal_angle = (
                compute_goal_angle(  ##compute orientation difference to reach goal
                    self._goal_pose.orientation, self.orientation
                )
            )
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computedenv
            initial_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)

            velocities = ctr_point_vel_from_agent_kinematics(
                initial_angular_vel,
                initial_velocity,
                number_ctrpt=self.ctr_pt_number,
                global_control_points=np.copy(global_control_points),
                environment_without_me=environment_without_me,
                priority=self.priority,
                DSM=True,
                reference_position=self.position,
                cutoff_gamma_obs=self.cutoff_gamma_obs,
            )

        elif version == "v1":
            velocities = compute_ctr_point_vel_from_obs_avoidance(
                number_ctrpt=self.ctr_pt_number,
                goal_pos_ctr_pts=goal_control_points,
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
                global_reference_position=self.position,
            )
            self.linear_velocity = linear_velocity
            self.angular_velocity = angular_velocity
            return

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
                ) = get_gamma_product_crowd(  # TODO: Done elsewhere, for efficiency maybe will need to be delete
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
                if len(list_critic_gammas_indx) > 0:
                    (
                        linear_velocity,
                        angular_velocity,
                    ) = agent_kinematics_from_ctr_point_vel(
                        velocities,
                        weights,
                        global_control_points=np.copy(global_control_points),
                        ctrpt_number=self.ctr_pt_number,
                        global_reference_position=self.position,
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
                        reference_position=self.position,
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

        self.linear_velocity, self.angular_velocity = agent_kinematics_from_ctr_point_vel(
            velocities,
            weights,
            global_control_points=np.copy(global_control_points),
            ctrpt_number=self.ctr_pt_number,
            global_reference_position=self.position,
        )

        
        self.apply_kinematic_constraints()

    def compute_metrics(self, dt):
        return
        # Compute distance
        if not self.converged:
            self.total_distance += LA.norm(self.linear_velocity) * dt
            self.time_conv += dt

        # compute proximity
        R = 3  # radius
        distance = []

        for obs in self.get_obstacles_without_me():
            distance.append(
                # get_distance_to_obtacle_surface(
                #     obstacle=obs,
                #     position=self.position,
                #     in_obstacle_frame=False,
                #     margin_absolut=self.margin_absolut,
                # )
                LA.norm(obs.position - self.position)
            )

        dmin = min(distance)
        # dmin = dmin if dmin < R else R
        # dmin = dmin if dmin > 0 else 0

        # self._proximity += dmin / R
        self._list_prox.append(
            dmin
        )  # Temporary metric used for the prox graph of the report, can be deleted

    def corner_case(self, mini_drag: str = "nodrag", version: str = "v1"):
        initial_velocity = np.zeros(2)
        environment_without_me = self.get_obstacles_without_me()
        # TODO : Make it a method to be called outside the class
        global_control_points = np.copy(global_control_points)

        weights = get_weight_of_control_points(
            global_control_points, environment_without_me
        )

        gamma_values = np.zeros(global_control_points.shape[1])
        for ii in range(global_control_points.shape[1]):
            gamma_values[ii] = get_gamma_product_crowd(
                global_control_points[:, ii], environment_without_me
            )

        velocities = np.zeros((self.dimension, self.ctr_pt_number))
        init_velocities = np.zeros((self.dimension, self.ctr_pt_number))

        if not self.static:
            # TODO : Do we want to enable rotation along other axis in the futur ?
            angular_vel = np.zeros((1, self.ctr_pt_number))

            # First we compute the initial velocity at the "center", ugly
            initial_velocity = self._dynamics.evaluate(self.position)

            initial_velocity = obs_avoidance_interpolation_moving(
                position=self.position,
                initial_velocity=initial_velocity,
                obs=environment_without_me,
                self_priority=self.priority,
            )

            initial_magnitude = LA.norm(initial_velocity)

            # Computing the weights of the angle to reach
            d = LA.norm(self.position - self._goal_pose.position)
            if mini_drag == "dragvel":
                w1_hat = self.virtual_drag
                w2_hat_max = 1000
                if LA.norm(initial_velocity) != 0:
                    w2_hat = (
                        self._dynamics.maximum_velocity / LA.norm(initial_velocity) - 1
                    )
                    if w2_hat > w2_hat_max:
                        w2_hat = w2_hat_max
                else:
                    w2_hat = w2_hat_max

                w1 = w1_hat / (w1_hat + w2_hat)
                w2 = 1 - w1
            elif mini_drag == "dragdist":
                kappa = self.virtual_drag
                w1 = 1 / 2 * (1 + np.tanh((d * kappa - 1.5 * kappa) / 2))
                w2 = 1 - w1

            elif mini_drag == "nodrag":
                w1 = 0
                w2 = 1
            else:
                print("Error in the name of the type of drag to use")
                w1 = 0
                w2 = 1

            # Direction (angle), of the linear_velocity in the global frame
            lin_vel_dir = np.arctan2(initial_velocity[1], initial_velocity[0])

            # Make the smallest rotation- the furniture has to pi symetric
            drag_angle = lin_vel_dir - self.orientation
            # Case where there is no symetry in the furniture
            if np.abs(drag_angle) > np.pi:
                drag_angle = -1 * (2 * np.pi - drag_angle)

            # Case where we consider for instance PI-symetry for the furniture
            if (
                np.abs(drag_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                drag_angle = lin_vel_dir - orientation_sym
                if drag_angle > np.pi / 2:
                    drag_angle = -1 * (2 * np.pi - drag_angle)

            goal_angle = self._goal_pose.orientation - self.orientation
            if np.abs(goal_angle) > np.pi:
                goal_angle = -1 * (2 * np.pi - goal_angle)

            if (
                np.abs(goal_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                goal_angle = self._goal_pose.orientation - orientation_sym
                if goal_angle > np.pi / 2:
                    goal_angle = -1 * (2 * np.pi - goal_angle)

            # TODO Very clunky : Rather make a function out of it
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computed
            initial_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)

            sign_project = np.zeros(self.ctr_pt_number)
            for ii in range(self.ctr_pt_number):
                # doing the cross product formula by "hand" than using the funct
                tang_vel = [
                    -initial_angular_vel * self._control_points[ii, 1],
                    initial_angular_vel * self._control_points[ii, 0],
                ]
                tang_vel = self.get_veloctity_in_global_frame(tang_vel)
                init_velocities[:, ii] = initial_velocity + tang_vel

                ctp = global_control_points[:, ii]
                velocities[:, ii] = obs_avoidance_interpolation_moving(
                    position=ctp,
                    initial_velocity=init_velocities[:, ii],
                    obs=environment_without_me,
                    self_priority=self.priority,
                )

                normal = np.array([1, 0])
                velocities_temp = self.get_velocity_in_local_frame(velocities[:, ii])
                sign_project[ii] = np.sign(np.dot(normal, velocities_temp))

                # plt.arrow(ctp[0], ctp[1], init_velocities[0, ii],
                #           init_velocities[1, ii], head_width=0.1, head_length=0.2, color='g')
                # plt.arrow(ctp[0], ctp[1], velocities[0, ii], velocities[1,
                #           ii], head_width=0.1, head_length=0.2, color='m')

            gamma_critic = 2
            magnitude = 1
            if np.sign(np.prod(sign_project)) < 0:
                if all(gamma_values < gamma_critic):
                    print("ANTI COLLSION")
                    magnitude = (
                        1 / (1 - gamma_critic) * (min(gamma_values) - gamma_critic)
                    )
                else:
                    magnitude = 1

            self.linear_velocity = np.sum(
                velocities * np.tile(weights, (self.dimension, 1)), axis=1
            )

            # normalization to the initial velocity
            self.linear_velocity = (
                initial_magnitude * self.linear_velocity / LA.norm(self.linear_velocity)
            ) * magnitude
            # plt.arrow(self.position[0], self.position[1], self.linear_velocity[0],
            #           self.linear_velocity[1], head_width=0.1, head_length=0.2, color='b')

            for ii in range(self.ctr_pt_number):
                angular_vel[0, ii] = weights[ii] * np.cross(
                    global_control_points[:, ii] - self._shape.center_position,
                    velocities[:, ii] - self.linear_velocity,
                )

            self.angular_velocity = np.sum(angular_vel) * magnitude

        else:
            self.linear_velocity = np.zeros(2)
            self.angular_velocity = 0.0


class Person(BaseAgent):
    def __init__(
        self,
        priority_value: float = 1,
        center_position: Optional[np.ndarray] = None,
        radius: float = 0.5,
        margin: float = 1,
        emergency_stop: bool = True,
        **kwargs,
    ) -> None:
        _shape = Ellipse(
            center_position=np.array(center_position),
            margin_absolut=margin,
            orientation=0,
            tail_effect=False,
            axes_length=np.array([radius, radius]),
        )

        super().__init__(
            shape=_shape,
            priority_value=priority_value,
            control_points=np.array([[0, 0]]),
            object_type=ObjectType.QOLO,
            **kwargs,
        )  # ALWAYS USE np.array([[0,0]]) and not np.array([0,0])

        self.maximum_velocity = 0.9
        self.maximum_acceleration = 2
        self.emergency_stop = emergency_stop
        self.gamma_stop = self.gamma_critic_min - 0.3
        self._dynamics = LinearSystem(
            attractor_position=self.position, maximum_velocity=self.maximum_velocity
        )

    @property
    def margin_absolut(self):
        return self._shape.margin_absolut

    def update_velocity(self, time_step, **kwargs):
        environment_without_me = self.get_obstacles_without_me()
        global_control_points = self.get_global_control_points()
        global_goal_control_points = self.get_goal_control_points()
        # Person only holds one control point, thus modulation is simplified
        # 0 hardcoded because we assume in person, we will have only one control point
        ctp = global_control_points[:, 0]
        # 0 hardcoded because we assume in person, we will have only one control point
        self._dynamics.attractor_position = global_goal_control_points[:, 0]

        if not self.static:
            velocity_old = self.linear_velocity

            if self.emergency_stop:
                gamma_indx, min_gamma = get_gamma_product_crowd(
                    self.position, environment_without_me
                )
                if min_gamma <= self.gamma_stop:
                    self.linear_velocity = np.zeros(2)
                    return

            initial_velocity = self._dynamics.evaluate(ctp)
            self.linear_velocity = obs_avoidance_interpolation_moving(
                position=ctp,
                initial_velocity=initial_velocity,
                obs=environment_without_me,
                self_priority=self.priority,
            )
            if LA.norm(self.linear_velocity) > self.maximum_velocity:
                # apply velocity constraints
                self.linear_velocity *= self.maximum_velocity / LA.norm(
                    self.linear_velocity
                )

            self.apply_acceleration_constraints(velocity_old, time_step)

        else:
            self.linear_velocity = np.zeros(2)

    def compute_metrics(self, delta_t):
        # compute the proximity
        R = 3  # radius
        distance = []

        for obs in self.get_obstacles_without_me():
            distance.append(
                # get_distance_to_obtacle_surface(
                #     obstacle=obs,
                #     position=self.position,
                #     in_obstacle_frame=False,
                #     margin_absolut=self.margin_absolut,
                # )
                LA.norm(obs.position - self.position)
            )

        dmin = min(distance)
        # dmin = dmin if dmin < R else R
        # dmin = dmin if dmin > 0 else 0
        # self._proximity += dmin / R
        self._list_prox.append(
            dmin
        )  # Temporary metric used for the prox graph of the report, can be deleted

    def apply_acceleration_constraints(self, velocity_old, time_step):
        # This function checks whether the difference in new computed kinematics and old kinematics exceeds the acceleration limits
        velocity_difference = self.linear_velocity - velocity_old
        velocity_difference_allowed = self.maximum_acceleration * time_step

        if LA.norm(velocity_difference) > velocity_difference_allowed:
            vel_correction = (
                velocity_difference
                / LA.norm(velocity_difference)
                * velocity_difference_allowed
            )
            self.linear_velocity = velocity_old + vel_correction

    # def get_distance_to_surface(
    #     self, position, in_obstacle_frame: bool = True, margin_absolut: float = None
    # ):
    #     self.get_point_on_surface(
    #         position=position,
    #         in_obstacle_frame=in_obstacle_frame,
    #         margin_absolut=margin_absolut,
    #     )
