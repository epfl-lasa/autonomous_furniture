from math import sqrt, cos, sin
import numpy as np
from numpy import linalg as LA
import warnings
from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import BaseContainer, ObstacleContainer
from vartools.dynamical_systems import DynamicalSystem, LinearSystem

from dynamic_obstacle_avoidance.obstacles import GammaType

from dynamic_obstacle_avoidance.avoidance.modulation import obs_avoidance_interpolation_moving


class FurnitureContainer(BaseContainer):
    def __getitem__(self, key):
        """List-like or dictionarry-like access to obstacle"""
        if isinstance(key, (str)):
            for ii in range(len(self._obstacle_list)):
                if self._obstacle_list[ii].name == key:
                    return self._obstacle_list[ii]
            raise ValueError("Obstacle <<{}>> not in list.".format(key))
        else:
            return self._obstacle_list[key]

    def __setitem__(self, key, value):
        # Is this useful?
        self._obstacle_list[key] = value

    def generate_obstacle_environment(self):
        obstacle_environment = ObstacleContainer()
        for value in self._obstacle_list:
            obstacle_environment.append(value.furniture_container)
        return obstacle_environment


class Furniture:
    def __init__(
            self,
            furniture_type: str = "furniture",
            num_control_points: int = 0,
            size=None,
            shape: str = "Cuboid",
            initial_position: np.ndarray = np.array([0, 0]),
            initial_orientation: float = 0.0,
            initial_speed: np.ndarray = np.array([0, 0]),
            goal_position: np.ndarray = np.array([0, 0]),
            goal_orientation: float = 0.0,
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

        self.num_control_points = num_control_points
        if self.num_control_points > 0:
            self.goal_position = goal_position
            self.goal_orientation = goal_orientation
            self.goal_container = self.generate_goal_container()
            self.rel_ctl_pts_pos, self.radius = self.calculate_relative_position()
            global_ctl_pos = self.relative2global(self.rel_ctl_pts_pos, self.goal_container)
            self.initial_dynamic = []
            for i in range(self.num_control_points):
                self.initial_dynamic.append(
                    LinearSystem(
                        attractor_position=global_ctl_pos[i],
                        maximum_velocity=1,
                        distance_decrease=0.3
                    )
                )

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
        div = self.max_ax / (self.num_control_points + 1)
        radius = sqrt(((self.min_ax / 2) ** 2) + (div ** 2))
        relative_control_point_pos = np.zeros((self.num_control_points, 2))

        for i in range(self.num_control_points):
            relative_control_point_pos[i, 0] = (div * (i + 1)) - (self.max_ax / 2)

        return relative_control_point_pos, radius

    def relative2global(self, relative_pos, obstacle):
        angle = obstacle.orientation
        obs_pos = obstacle.center_position
        global_pos = np.zeros_like(relative_pos)
        rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

        for i in range(relative_pos.shape[0]):
            rot_rel_pos = np.dot(rot, relative_pos[i, :])
            global_pos[i, :] = obs_pos + rot_rel_pos

        return global_pos

    def global2relative(self, global_pos, obstacle):
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
        temp_env = self.furniture_env[0:obs_index] + self.furniture_env[obs_index + 1:]
        return temp_env

    def get_gamma_product_furniture(self, position, env, gamma_type=GammaType.EUCLEDIAN):
        if not len(env):
            # Very large number
            return 1e20

        gamma_list = np.zeros(len(env))
        for ii, obs in enumerate(env):
            # gamma_type needs to be implemented for all obstacles
            gamma_list[ii] = obs.furniture_container.get_gamma(
                position, in_global_frame=True, gamma_type=gamma_type
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
            gamma_values[cp] = self.get_gamma_product_furniture(control_points[cp, :], env)

        return gamma_values

    @staticmethod
    def get_weight_from_gamma(gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5):
        weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
        weights = weights / frac_gamma_nth
        weights = 1.0 / weights
        weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
        weights = weights / n_points
        return weights

    def get_influence_weight_at_points(self, control_points, cutoff_gamma=5):
        # TODO
        ctl_weight_list = []
        for index, furniture in enumerate(self.furniture_env):
            if furniture.num_control_points == 0:
                break
            temp_env = self.env_slicer(index)
            gamma_values = self.get_gamma_at_point(control_points[index], index, temp_env)

            ctl_point_weight = np.zeros(gamma_values.shape)
            ind_nonzero = gamma_values < cutoff_gamma
            if not any(ind_nonzero):
                # ctl_point_weight[-1] = 1
                ctl_point_weight = np.full(gamma_values.shape, 1/furniture.num_control_points)
            # for index in range(len(gamma_values)):
            ctl_point_weight[ind_nonzero] = self.get_weight_from_gamma(
                gamma_values[ind_nonzero],
                cutoff_gamma=cutoff_gamma,
                n_points=furniture.num_control_points
            )

            ctl_point_weight_sum = np.sum(ctl_point_weight)
            if ctl_point_weight_sum > 1:
                ctl_point_weight = ctl_point_weight / ctl_point_weight_sum
            else:
                ctl_point_weight[-1] += 1 - ctl_point_weight_sum

            ctl_weight_list.append(ctl_point_weight)

        return ctl_weight_list

    def evaluate_furniture(self, position: np.ndarray, selected_furniture) -> np.ndarray:
        initial_velocities = np.zeros((self.furniture_env[selected_furniture].num_control_points, self.dim))
        for ctl_pt in range(self.furniture_env[selected_furniture].num_control_points):
            initial_velocities[ctl_pt, :] = self.furniture_env[selected_furniture].initial_dynamic[ctl_pt].evaluate(position[ctl_pt, :])
        temp_env = self.env_slicer(selected_furniture)
        updated_velocities = np.zeros((self.furniture_env[selected_furniture].num_control_points, self.dim))
        for ctl_pt in range(self.furniture_env[selected_furniture].num_control_points):
            updated_velocities[ctl_pt, :] = self.avoid_furniture(position[ctl_pt, :], initial_velocities[ctl_pt, :], temp_env)
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

    def avoid_furniture(self, position: np.ndarray, initial_velocity: np.ndarray, env, const_speed: bool = True) -> np.ndarray:
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

    # def get_attractor_position(self, control_point):
    #     return self.initial_dynamics[control_point].attractor_position

    # def set_attractor_position(self, position: np.ndarray, control_point):
    #     self.initial_dynamics[control_point].attractor_position = position
