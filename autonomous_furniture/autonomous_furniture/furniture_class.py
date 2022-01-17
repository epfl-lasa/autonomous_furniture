import math
import numpy as np
from numpy import linalg as LA
import warnings
from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.obstacles import GammaType

from dynamic_obstacle_avoidance.avoidance.modulation import obs_avoidance_interpolation_moving


class Furniture:
    def __init__(
            self,
            is_controllable: bool,
            num_control_points: int,
            size,
            shape: str,
            initial_position: np.ndarray,
            initial_orientation: float,
            initial_speed: np.ndarray,
            goal_position: np.ndarray,
            goal_orientation: float,
    ):
        self.is_controllable = is_controllable
        self.size = size
        self.shape = shape
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.initial_speed = initial_speed

        if self.size is not None:
            self.max_ax = max(self.size)
            self.min_ax = min(self.size)

        self.furniture_container = self.generate_furniture_container()

        if self.is_controllable:
            self.num_control_points = num_control_points
            self.goal_position = goal_position
            self.goal_orientation = goal_orientation
            self.goal_container = self.generate_goal_container()
            self.rel_ctl_pts_pos, self.radius = self.calculate_relative_position()

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
        radius = math.sqrt(((self.min_ax / 2) ** 2) + (div ** 2))
        relative_control_point_pos = np.zeros((self.num_control_points, 2))

        for i in range(self.num_control_points):
            relative_control_point_pos[i, 0] = (div * (i + 1)) - (self.max_ax / 2)

        return relative_control_point_pos, radius

    def relative2global(self, relative_pos, obstacle):
        angle = obstacle.orientation
        obs_pos = obstacle.center_position
        global_pos = np.zeros_like(relative_pos)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

        for i in range(relative_pos.shape[0]):
            rot_rel_pos = np.dot(rot, relative_pos[i, :])
            global_pos[i, :] = obs_pos + rot_rel_pos

        return global_pos

    def global2relative(self, global_pos, obstacle):
        angle = -1 * obstacle.orientation
        obs_pos = obstacle.center_position
        relative_pos = np.zeros_like(global_pos)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

        for i in range(global_pos.shape[0]):
            rel_pos_pre_rot = global_pos[i, :] - obs_pos
            relative_pos[i, :] = np.dot(rot, rel_pos_pre_rot)

        return relative_pos

    def get_num_stl_pts(self):
        return self.num_control_points


class FurnitureDynamics:
    def __init__(
            self,
            initial_dynamics: DynamicalSystem,
            environment,
            maximum_speed: float = None,
    ):
        self.initial_dynamics = initial_dynamics
        self.environment = environment
        self.maximum_speed = maximum_speed

    def env_slicer(self, obs_index):
        temp_env = self.environment[0:obs_index] + self.environment[obs_index + 1:]
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
        num_ctl_pts = self.environment[obs_eval].num_control_points
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
        last_ctl_pts = 0
        for index, furniture in enumerate(self.environment):
            if not furniture.is_controllable:
                break
            temp_env = self.env_slicer(index)
            gamma_values = self.get_gamma_at_point(control_points[last_ctl_pts:last_ctl_pts+furniture.num_control_points], index, temp_env)
            last_ctl_pts += furniture.num_control_points

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

    def evaluate_furniture(self, position: np.ndarray, selected_ctl_pt, env) -> np.ndarray:
        """DynamicalSystem compatible 'evaluate' method that returns the velocity at
        a given input position."""
        return self.compute_dynamics_furniture(position, selected_ctl_pt, env)

    def compute_dynamics_furniture(self, position: np.ndarray, selected_ctl_pt, env) -> np.ndarray:
        """DynamicalSystem compatible 'compute_dynamics' method that returns the
        velocity at a given input position."""
        initial_velocity = self.initial_dynamics[selected_ctl_pt].evaluate(position)
        return self.avoid_furniture(position=position, initial_velocity=initial_velocity, env=env)

    def avoid_furniture(self, position: np.ndarray, initial_velocity: np.ndarray, env, const_speed: bool = True) -> np.ndarray:
        obs = ObstacleContainer()
        for furniture in env:
            obs.append(furniture.furniture_container)

        vel = obs_avoidance_interpolation_moving(
            position=position, initial_velocity=initial_velocity, obs=obs
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

    def get_attractor_position(self, control_point):
        return self.initial_dynamics[control_point].attractor_position

    def set_attractor_position(self, position: np.ndarray, control_point):
        self.initial_dynamics[control_point].attractor_position = position
