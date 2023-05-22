import warnings
from math import sqrt, cos, sin

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import DynamicalSystem, LinearSystem

# from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.containers import BaseContainer, ObstacleContainer

# from dynamic_obstacle_avoidance.obstacles import GammaType
from dynamic_obstacle_avoidance.avoidance.modulation import (
    obs_avoidance_interpolation_moving,
)


class FurnitureContainer(BaseContainer):
    def __getitem__(self, key):
        """List-like or dictionarry-like access to obstacle"""
        if isinstance(key, str):
            for ii in range(len(self._obstacle_list)):
                if self._obstacle_list[ii].name == key:
                    return self._obstacle_list[ii]
            raise ValueError("Obstacle <<{}>> not in list.".format(key))
        else:
            return self._obstacle_list[key]

    def __setitem__(self, key, value):
        # Is this useful?
        self._obstacle_list[key] = value

    # def append(self, value):  # Compatibility with normal list.
    #     self._obstacle_list.append(value)

    def generate_obstacle_environment(self):
        obstacle_environment = ObstacleContainer()
        for value in self._obstacle_list:
            obstacle_environment.append(value.furniture_container)
        return obstacle_environment

    def extract_margin(self):
        margin_list = []
        for furniture in self._obstacle_list:
            if furniture.furniture_type == "furniture":
                margin_list.append(furniture.radius)
        return margin_list

    def assign_margin(self):
        adjusting_factor = 1.2
        margin_list = self.extract_margin()
        index_max = np.argmax(margin_list)
        reduced_margin_list = margin_list[:]
        reduced_margin_list.pop(index_max)
        for furniture in self._obstacle_list:
            if furniture.furniture_type == "person":
                furniture.furniture_container.margin_absolut = (
                    max(margin_list) / adjusting_factor
                )
            elif furniture.furniture_type == "furniture":
                if furniture.radius != margin_list[index_max]:
                    furniture.furniture_container.margin_absolut = (
                        margin_list[index_max] / adjusting_factor
                    )
                else:
                    furniture.furniture_container.margin_absolut = (
                        max(reduced_margin_list) / adjusting_factor
                    )


class Furniture:
    def __init__(
        self,
        furniture_type: str = "furniture",
        num_control_points=None,
        size=None,
        shape: str = "Cuboid",
        initial_position: np.ndarray = np.array([0, 0]),
        initial_orientation: float = 0.0,
        initial_speed: np.ndarray = np.array([0, 0]),
        goal_position: np.ndarray = np.array([0, 0]),
        goal_orientation: float = 0.0,
        parking_zone_position=None,
        parking_zone_orientation=None,
    ):
        # if not "person" or "furniture" in furniture_type:
        #     raise ValueError("Can only be person or furniture")
        self.furniture_type = furniture_type
        self.size = size
        self.shape = shape
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.initial_speed = initial_speed

        if self.size is not None:
            self.max_ax = max(self.size)
            self.min_ax = min(self.size)

        self.furniture_container = self.generate_furniture_container()

        if num_control_points is None:
            self.num_control_points = 0
        if isinstance(num_control_points, int):
            self.num_control_points = num_control_points
            self.control_points_axis_0 = num_control_points
            self.control_points_axis_1 = 1
        elif isinstance(num_control_points, tuple):
            self.num_control_points = num_control_points[0] * num_control_points[1]
            self.control_points_axis_0 = num_control_points[0]
            self.control_points_axis_1 = num_control_points[1]

        if self.num_control_points > 0:
            self.goal_position = goal_position
            self.goal_orientation = goal_orientation
            self.goal_container = self.generate_goal_container()
            self.rel_ctl_pts_pos, self.radius = self.calculate_relative_position()
            global_ctl_pos = self.relative2global(
                self.rel_ctl_pts_pos, self.goal_container
            )
            self.initial_dynamic = []
            for i in range(self.num_control_points):
                self.initial_dynamic.append(
                    LinearSystem(
                        attractor_position=global_ctl_pos[i],
                        maximum_velocity=1,
                        distance_decrease=0.3,
                    )
                )
            if parking_zone_position is None:
                self.parking_zone_position = initial_position
            else:
                self.parking_zone_position = parking_zone_position
            if parking_zone_orientation is None:
                self.parking_zone_orientation = initial_orientation
            else:
                self.parking_zone_orientation = parking_zone_orientation
            self.attractor_state = "idle"

    def generate_furniture_container(self):
        if self.shape == "Ellipse":
            return Ellipse(
                axes_length=[self.max_ax, self.min_ax],
                center_position=self.initial_position,
                margin_absolut=0,
                orientation=self.initial_orientation,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=self.initial_speed,
            )
        elif self.shape == "Cuboid":
            return Cuboid(
                axes_length=[self.max_ax, self.min_ax],
                center_position=self.initial_position,
                margin_absolut=0,
                orientation=self.initial_orientation,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=self.initial_speed,
            )

    def generate_goal_container(self):
        return Cuboid(
            axes_length=[self.max_ax, self.min_ax],
            center_position=self.goal_position,
            margin_absolut=0,
            orientation=self.goal_orientation,
            tail_effect=False,
            repulsion_coeff=1,
        )

    def calculate_relative_position(self):
        div_max_ax = self.max_ax / (self.control_points_axis_0 + 1)
        div_min_ax = self.min_ax / (self.control_points_axis_1 + 1)
        radius = sqrt((div_max_ax**2) + (div_min_ax**2))
        relative_control_point_pos = np.zeros((self.num_control_points, 2))

        k = 0
        for j in range(self.control_points_axis_1):
            for i in range(self.control_points_axis_0):
                relative_control_point_pos[k, 0] = (div_max_ax * (i + 1)) - (
                    self.max_ax / 2
                )
                relative_control_point_pos[k, 1] = (div_min_ax * (j + 1)) - (
                    self.min_ax / 2
                )
                k += 1

        return relative_control_point_pos, radius

    @staticmethod
    def relative2global(relative_pos, obstacle):
        angle = obstacle.orientation
        obs_pos = obstacle.center_position
        global_pos = np.zeros_like(relative_pos)
        rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

        for i in range(relative_pos.shape[0]):
            rot_rel_pos = np.dot(rot, relative_pos[i, :])
            global_pos[i, :] = obs_pos + rot_rel_pos

        return global_pos

    @staticmethod
    def global2relative(global_pos, obstacle):
        angle = -1 * obstacle.orientation
        obs_pos = obstacle.center_position
        relative_pos = np.zeros_like(global_pos)
        rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

        for i in range(global_pos.shape[0]):
            rel_pos_pre_rot = global_pos[i, :] - obs_pos
            relative_pos[i, :] = np.dot(rot, rel_pos_pre_rot)

        return relative_pos

    def get_num_ctl_pts(self):
        return self.num_control_points

    def get_initial_dynamic(self):
        return self.initial_dynamic

    def get_margin(self):
        return self.furniture_container.margin_absolut

    def get_furniture_pos(self):
        return self.furniture_container.center_position

    def get_furniture_orientation(self):
        return self.furniture_container.orientation

    def set_furniture_pos(self, pos: np.ndarray):
        self.furniture_container.center_position = pos

    def set_furniture_orientation(self, ori: float):
        self.furniture_container.orientation = ori

    def get_goal_pos(self):
        return self.goal_container.center_position

    def get_goal_orientation(self):
        return self.goal_container.orientation

    def set_goal_pos(self, pos: np.ndarray):
        self.goal_container.center_position = pos

    def set_goal_orientation(self, ori: float):
        self.goal_container.orientation = ori


def get_weight_from_gamma(
    gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5
):
    weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
    weights = weights / frac_gamma_nth
    weights = 1.0 / weights
    weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
    weights = weights / n_points
    return weights


class FurnitureDynamics:
    dim = 2

    def __init__(
        self,
        # initial_dynamics: DynamicalSystem,
        # environment,
        furniture_env: FurnitureContainer,
        maximum_speed: float = None,
    ):
        # self.initial_dynamics = initial_dynamics
        self.environment = None
        self.furniture_env = furniture_env
        self.maximum_speed = maximum_speed

    def update_environment(self, env):
        self.environment = ObstacleContainer()
        for furniture in env:
            self.environment.append(furniture.furniture_container)

    def env_slicer(self, obs_index):
        temp_env = self.furniture_env[0:obs_index] + self.furniture_env[obs_index + 1 :]
        return temp_env

    @staticmethod
    def get_gamma_product_furniture(
        position: np.ndarray,
        env: ObstacleContainer,
        # gamma_type: GammaType = GammaType.EUCLEDIAN,
    ):
        if not len(env):
            # Very large number
            return 1e20

        gamma_list = np.zeros(len(env))
        for ii, obs in enumerate(env):
            # gamma_type needs to be implemented for all obstacles
            # position, in_global_frame=True, gamma_type=gamma_type
            gamma_list[ii] = obs.furniture_container.get_gamma(
                position, in_global_frame=True
            )

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            warnings.warn("Collision detected.")
            # breakpoint()
            return 0

        # gamma = np.prod(gamma_list-1)**(1.0/n_obs) + 1
        gamma = np.min(gamma_list)

        if np.isnan(gamma):
            breakpoint()
        return gamma

    def get_gamma_at_point(self, control_points, obs_eval, env):
        num_ctl_pts = self.furniture_env[obs_eval].num_control_points
        gamma_values = np.zeros(num_ctl_pts)

        for cp in range(num_ctl_pts):
            gamma_values[cp] = self.get_gamma_product_furniture(
                control_points[cp, :], env
            )

        return gamma_values

    # @staticmethod
    # def get_weight_from_gamma(gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5):
    #     weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
    #     weights = weights / frac_gamma_nth
    #     weights = 1.0 / weights
    #     weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
    #     weights = weights / n_points
    #     return weights

    def get_influence_weight_at_points(self, control_points, cutoff_gamma=5):
        ctl_weight_list = []
        for index, furniture in enumerate(self.furniture_env):
            if furniture.num_control_points == 0:
                break
            temp_env = self.env_slicer(index)
            gamma_values = self.get_gamma_at_point(
                control_points[index], index, temp_env
            )

            ctl_point_weight = np.zeros(gamma_values.shape)
            ind_nonzero = gamma_values < cutoff_gamma
            if not any(ind_nonzero):
                # ctl_point_weight[-1] = 1
                ctl_point_weight = np.full(
                    gamma_values.shape, 1 / furniture.num_control_points
                )
            # for index in range(len(gamma_values)):
            ctl_point_weight[ind_nonzero] = get_weight_from_gamma(
                gamma_values[ind_nonzero],
                cutoff_gamma=cutoff_gamma,
                n_points=furniture.num_control_points,
            )

            ctl_point_weight_sum = np.sum(ctl_point_weight)
            if ctl_point_weight_sum > 1:
                ctl_point_weight = ctl_point_weight / ctl_point_weight_sum
            else:
                ctl_point_weight[-1] += 1 - ctl_point_weight_sum

            ctl_weight_list.append(ctl_point_weight)

        return ctl_weight_list

    def evaluate_furniture(
        self, position: np.ndarray, selected_furniture
    ) -> np.ndarray:
        initial_velocities = np.zeros(
            (self.furniture_env[selected_furniture].num_control_points, self.dim)
        )
        for ctl_pt in range(self.furniture_env[selected_furniture].num_control_points):
            initial_velocities[ctl_pt, :] = (
                self.furniture_env[selected_furniture]
                .initial_dynamic[ctl_pt]
                .evaluate(position[ctl_pt, :])
            )
        temp_env = self.env_slicer(selected_furniture)
        updated_velocities = np.zeros(
            (self.furniture_env[selected_furniture].num_control_points, self.dim)
        )
        for ctl_pt in range(self.furniture_env[selected_furniture].num_control_points):
            updated_velocities[ctl_pt, :] = self.avoid_furniture(
                position[ctl_pt, :], initial_velocities[ctl_pt, :], temp_env
            )
        return updated_velocities

    # def evaluate_furniture(self, position: np.ndarray, selected_ctl_pt, env) -> np.ndarray:
    #     """DynamicalSystem compatible 'evaluate' method that returns the velocity at
    #     a given input position."""
    #     return self.compute_dynamics_furniture(position, selected_ctl_pt, env)

    # def compute_dynamics_furniture(self, position: np.ndarray, selected_ctl_pt, env) -> np.ndarray:
    #     """DynamicalSystem compatible 'compute_dynamics' method that returns the
    #     velocity at a given input position."""
    #     initial_velocity = self.initial_dynamics[selected_ctl_pt].evaluate(position)
    #     return self.avoid_furniture(position=position, initial_velocity=initial_velocity, env=env)

    def avoid_furniture(
        self,
        position: np.ndarray,
        initial_velocity: np.ndarray,
        env,
        const_speed: bool = True,
    ) -> np.ndarray:
        # obs = ObstacleContainer()
        # for furniture in env:
        #     obs.append(furniture.furniture_container)

        self.update_environment(env)

        vel = obs_avoidance_interpolation_moving(
            position=position, initial_velocity=initial_velocity, obs=self.environment
        )
        # Adapt speed if desired
        if const_speed:
            vel_mag = LA.norm(vel)
            if vel_mag:
                vel = vel / vel_mag * LA.norm(initial_velocity)

        elif self.maximum_speed is not None:
            vel_mag = LA.norm(vel)
            if vel_mag > self.maximum_speed:
                vel = vel / vel_mag * self.maximum_speed

        return vel

    def get_attractor_position(self):
        attra_pos = []
        for ii, furniture in enumerate(self.furniture_env):
            attra_pos.append(furniture.initial_dynamic[ii].attractor_position)
        return attra_pos

    def set_attractor_position(self, position_list, selected_furniture):
        for ii, position in enumerate(position_list):
            self.furniture_env[selected_furniture].initial_dynamic[
                ii
            ].attractor_position = position


class FurnitureAttractorDynamics:
    min_dist = 1
    dim = 2

    def __init__(
        self,
        furniture_env: FurnitureContainer,
        cutoff_distance=5,
        max_repulsion=1,
    ):
        self.furniture_env = furniture_env
        self.cutoff_distance = cutoff_distance
        self.max_repulsion = max_repulsion
        self.lambda_p = 10
        self.attractor_state = []
        self.reduced_env = self.furniture_env[:]
        for index, furniture in enumerate(self.furniture_env):
            if furniture.num_control_points > 0:
                self.attractor_state.append(["idle"] * furniture.num_control_points)
            if furniture.furniture_type == "person":
                self.reduced_env.pop(index)
                self.person_index = index

    @staticmethod
    def generate_temp_environment(env):
        environment = ObstacleContainer()
        for furniture in env:
            environment.append(furniture.furniture_container)
        return environment

    def evaluate_attractor(
        self, position: np.ndarray, selected_attractor, selected_furniture
    ) -> np.ndarray:
        direction_vect = (
            position
            - self.furniture_env[self.person_index].furniture_container.center_position
        )
        distance_vect = self.furniture_env[
            self.person_index
        ].furniture_container.get_gamma(position, in_global_frame=True)

        if distance_vect < 1:
            print("attractor got run over :'(")
            # TODO: no theoretical value
            return np.zeros(self.dim)

        temp_env = (
            self.reduced_env[:selected_furniture]
            + self.reduced_env[selected_furniture + 1 :]
        )
        unit_direction_vect = direction_vect / np.linalg.norm(direction_vect)
        unit_person_linear_velocity = self.furniture_env[
            self.person_index
        ].furniture_container.linear_velocity / np.linalg.norm(
            self.furniture_env[self.person_index].furniture_container.linear_velocity
        )
        max_repulsion = np.dot(unit_direction_vect, unit_person_linear_velocity)

        if max_repulsion <= 0:
            self.attractor_state[selected_furniture][selected_attractor] = "idle"
            return np.zeros(self.dim)
        else:
            self.attractor_state[selected_furniture][selected_attractor] = "avoid"
            # pass

        slope = (-max_repulsion) / (self.cutoff_distance - self.min_dist)
        offset = max_repulsion - (slope * self.min_dist)
        repulsion_magnitude = (slope * distance_vect) + offset

        if repulsion_magnitude <= 0.0:
            return np.zeros(self.dim)

        base_attractor_velocity = repulsion_magnitude * unit_direction_vect
        eigenvector_person = np.array(
            [-unit_person_linear_velocity[1], unit_person_linear_velocity[0]]
        )
        e_matrix = np.array([unit_person_linear_velocity, eigenvector_person])
        a_matrix = np.eye(self.dim)
        a_matrix[1] *= self.lambda_p
        attractor_velocity = e_matrix.T @ a_matrix @ e_matrix @ base_attractor_velocity
        mod_attractor_velocity = self.avoid_attractor(
            position, attractor_velocity, temp_env
        )

        # if self.furniture_env[selected_furniture].attractor_state == "idle":
        #     unit_direction_vect = direction_vect / np.linalg.norm(direction_vect)
        #     unit_person_linear_velocity = self.furniture_env[
        #                                       self.person_index].furniture_container.linear_velocity / np.linalg.norm(
        #         self.furniture_env[self.person_index].furniture_container.linear_velocity)
        #     max_repulsion = np.dot(unit_direction_vect, unit_person_linear_velocity)
        #
        #     if max_repulsion <= 0:
        #         self.furniture_env[selected_furniture].attractor_state = "idle"
        #         return np.zeros(self.dim)
        #
        #     slope = (-max_repulsion) / (self.cutoff_distance - self.min_dist)
        #     offset = max_repulsion - (slope * self.min_dist)
        #     repulsion_magnitude = (slope * distance_vect) + offset
        #
        #     if repulsion_magnitude <= 0.:
        #         self.furniture_env[selected_furniture].attractor_state = "idle"
        #         return np.zeros(self.dim)
        #     else:
        #         print("\n")
        #         print(max_repulsion)
        #         print("\n")
        #         print(repulsion_magnitude)
        #         print("\n")
        #         self.furniture_env[selected_furniture].attractor_state = "avoid"
        #
        #     base_attractor_velocity = repulsion_magnitude * unit_direction_vect
        #     eigenvector_person = np.array([-unit_person_linear_velocity[1], unit_person_linear_velocity[0]])
        #     e_matrix = np.array([unit_person_linear_velocity, eigenvector_person])
        #     a_matrix = np.eye(self.dim)
        #     a_matrix[1] *= self.lambda_p
        #     attractor_velocity = e_matrix.T @ a_matrix @ e_matrix @ base_attractor_velocity
        #     mod_attractor_velocity = self.avoid_attractor(position, attractor_velocity, temp_env)
        #
        # elif self.furniture_env[selected_furniture].attractor_state == "avoid":
        #     unit_direction_vect = direction_vect / np.linalg.norm(direction_vect)
        #     unit_person_linear_velocity = self.furniture_env[self.person_index].furniture_container.linear_velocity / np.linalg.norm(self.furniture_env[self.person_index].furniture_container.linear_velocity)
        #     max_repulsion = np.dot(unit_direction_vect, unit_person_linear_velocity)
        #
        #     if max_repulsion < 0.:
        #         print("\n")
        #         print(max_repulsion)
        #         print("\n")
        #         self.furniture_env[selected_furniture].attractor_state = "regroup"
        #         return np.zeros(self.dim)
        #
        #     slope = (-max_repulsion) / (self.cutoff_distance - self.min_dist)
        #     offset = max_repulsion - (slope * self.min_dist)
        #     repulsion_magnitude = (slope * distance_vect) + offset
        #
        #     if repulsion_magnitude <= 0.:
        #         print("\n")
        #         print(repulsion_magnitude)
        #         print("\n")
        #         print("bro why is repulsion magnitude different ?")
        #         print("\n")
        #         self.furniture_env[selected_furniture].attractor_state = "regroup"
        #         return np.zeros(self.dim)
        #
        #     base_attractor_velocity = repulsion_magnitude * unit_direction_vect
        #     eigenvector_person = np.array([-unit_person_linear_velocity[1], unit_person_linear_velocity[0]])
        #     e_matrix = np.array([unit_person_linear_velocity, eigenvector_person])
        #     a_matrix = np.eye(self.dim)
        #     a_matrix[1] *= self.lambda_p
        #     attractor_velocity = e_matrix.T @ a_matrix @ e_matrix @ base_attractor_velocity
        #     mod_attractor_velocity = self.avoid_attractor(position, attractor_velocity, temp_env)
        #
        # elif self.furniture_env[selected_furniture].attractor_state == "regroup":
        #     mod_attractor_velocity = np.zeros(self.dim)
        #     # TODO: add condition when v_f and omega_f == 0
        #
        # else:
        #     mod_attractor_velocity = np.zeros(self.dim)

        return mod_attractor_velocity

    def avoid_attractor(
        self,
        position: np.ndarray,
        initial_velocity: np.ndarray,
        furn_env,
        const_speed: bool = True,
    ):
        env = self.generate_temp_environment(furn_env)
        vel = obs_avoidance_interpolation_moving(
            position=position, initial_velocity=initial_velocity, obs=env
        )

        if const_speed:
            vel_mag = LA.norm(vel)
            if vel_mag:
                vel = vel / vel_mag * LA.norm(initial_velocity)

        return vel

    def set_lambda(self, lambda_p):
        self.lambda_p = lambda_p

    @staticmethod
    def get_gamma_product_attractor(
        position,
        env,
        # gamma_type=GammaType.EUCLEDIAN
    ):
        if not len(env):
            # Very large number
            return 1e20
        gamma_list = np.zeros(len(env))
        for ii, furniture in enumerate(env):
            # gamma_type needs to be implemented for all obstacles
            gamma_list[ii] = furniture.furniture_container.get_gamma(
                position, in_global_frame=True
            )

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            warnings.warn("Collision detected.")
            # breakpoint()
            return 0

        gamma = np.min(gamma_list)

        if np.isnan(gamma):
            breakpoint()
        return gamma

    def get_weights_attractors(self, attractor_points, selected_furniture):
        num_attractors = self.furniture_env[selected_furniture].num_control_points

        temp_env = (
            self.furniture_env[:selected_furniture]
            + self.furniture_env[selected_furniture + 1 :]
        )

        gamma_list = np.zeros(num_attractors)
        # for the moment only take into account the person/qolo/agent
        for ii in range(num_attractors):
            gamma_list[ii] = self.get_gamma_product_attractor(
                attractor_points[ii, :], temp_env
            )

        attractor_weights = np.zeros(gamma_list.shape)
        ind_nonzero = gamma_list < self.cutoff_distance
        if not any(ind_nonzero):
            attractor_weights = np.full(num_attractors, 1 / num_attractors)
            return attractor_weights

        attractor_weights[ind_nonzero] = get_weight_from_gamma(
            gamma_list[ind_nonzero],
            cutoff_gamma=self.cutoff_distance,
            n_points=num_attractors,
        )

        attractor_weights_sum = np.sum(attractor_weights)
        if attractor_weights_sum > 1:
            attractor_weights = attractor_weights / attractor_weights_sum
        else:
            attractor_weights[-1] = 1 - attractor_weights_sum

        return attractor_weights

    def get_goal_velocity(
        self, attractor_pos, attractor_velocities, attractor_weights, selected_furniture
    ):
        num_attractors = self.furniture_env[selected_furniture].num_control_points
        lin_vel = np.zeros(2)
        for ii in range(num_attractors):
            lin_vel += attractor_weights[ii] * attractor_velocities[ii, :]

        angular_vel = np.zeros(num_attractors)
        for attractor in range(num_attractors):
            angular_vel[attractor] = attractor_weights[attractor] * np.cross(
                (
                    self.furniture_env[
                        selected_furniture
                    ].furniture_container.center_position
                    - attractor_pos[attractor, :]
                ),
                (attractor_velocities[attractor, :] - lin_vel),
            )
        angular_vel_sum = angular_vel.sum()

        return lin_vel, angular_vel_sum

    def evaluate_furniture_attractor(self, position_list, selected_furniture):
        attractor_velocities = np.zeros(
            (self.furniture_env[selected_furniture].num_control_points, self.dim)
        )

        for ii, attractor in enumerate(position_list):
            attractor_velocities[ii, :] = self.evaluate_attractor(
                attractor, ii, selected_furniture
            )
            # print(f"state of furniture {selected_furniture} and attractor {ii}: {self.attractor_state[selected_furniture][ii]} \n")

        attractor_weights = self.get_weights_attractors(
            position_list, selected_furniture
        )
        goal_velocity, goal_rotation = self.get_goal_velocity(
            position_list, attractor_velocities, attractor_weights, selected_furniture
        )
        self.set_attractor_state(selected_furniture, goal_velocity, goal_rotation)

        return goal_velocity, goal_rotation

    def set_attractor_state(self, selected_furniture, goal_velocity, goal_rotation):
        current_state = self.furniture_env[selected_furniture].attractor_state
        if "avoid" in self.attractor_state[selected_furniture]:
            self.furniture_env[selected_furniture].attractor_state = "avoid"
        elif (
            "avoid" not in self.attractor_state[selected_furniture]
            and current_state == "avoid"
        ):
            self.furniture_env[selected_furniture].attractor_state = "regroup"
        elif current_state == "regroup":
            if np.sum(np.abs(goal_velocity)) < 1e-2 and np.abs(goal_rotation) < 1e-2:
                self.furniture_env[selected_furniture].attractor_state = "idle"
