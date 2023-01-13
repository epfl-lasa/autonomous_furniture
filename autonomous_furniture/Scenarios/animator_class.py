from math import pi, cos, sin, sqrt

import numpy as np 

import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

class DynamicalSystemAnimation(Animator):
    dim = 2

    def setup(
        self,
        initial_dynamics,
        obstacle_environment,
        start_position=None,
        walls=False,
        x_lim=None,
        y_lim=None,
    ):
        num_obs = len(obstacle_environment)

        if y_lim is None:
            y_lim = [-3., 3.]
        if x_lim is None:
            x_lim = [-3., 3.]
        if start_position is None:
            start_position = np.zeros((num_obs, self.dim))
        if walls is True:
            walls_center_position = np.array([[0., y_lim[0]], [x_lim[0], 0.], [0., y_lim[1]], [x_lim[1], 0.]])
            x_length = x_lim[1] - x_lim[0]
            y_length = y_lim[1] - y_lim[0]
            wall_margin = obstacle_environment[-1].margin_absolut
            walls_cont = [
                Cuboid(
                    axes_length=[x_length, 0.1],  # [x_length, y_length],
                    center_position=walls_center_position[0],  # np.array([0., 0.]),
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                    is_boundary=False,
                ),
                Cuboid(
                    axes_length=[0.1, y_length],
                    center_position=walls_center_position[1],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                ),
                Cuboid(
                    axes_length=[x_length, 0.1],
                    center_position=walls_center_position[2],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                ),
                Cuboid(
                    axes_length=[0.1, y_length],
                    center_position=walls_center_position[3],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                )
            ]
            new_obstacle_environment = ObstacleContainer()
            new_obstacle_environment.append(
                Cuboid(
                    axes_length=[4.1, 5.2],
                    center_position=np.array([0., 0.]),
                    margin_absolut=0.6,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                )
            )
            new_obstacle_environment = new_obstacle_environment + walls_cont
        else:
            new_obstacle_environment = ObstacleContainer()
            new_obstacle_environment.append(
                Cuboid(
                    axes_length=[4.1, 5.2],
                    center_position=np.array([0., 0.]),
                    margin_absolut=0.6,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                )
            )

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.real_obstacle_environment = obstacle_environment
        self.obstacle_environment = new_obstacle_environment
        self.initial_dynamics = initial_dynamics

        self.dynamic_avoider = DynamicModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            environment=self.obstacle_environment,
        )

        self.position_list = np.zeros((self.dim, self.it_max))
        self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        # Here come the main calculation part
        velocity = self.dynamic_avoider.evaluate(self.position_list[:, ii])
        self.position_list[:, ii + 1] = (
            velocity * self.dt_simulation + self.position_list[:, ii]
        )
        # print(

        # Update obstacles
        self.obstacle_environment.do_velocity_step(delta_time=self.dt_simulation)
        self.real_obstacle_environment[-1].center_position = self.position_list[:, ii + 1]

        self.ax.clear()

        # Drawing and adjusting of the axis
        self.ax.plot(
            self.position_list[0, :ii + 1],
            self.position_list[1, :ii + 1],
            ":",
            color="#135e08"
        )
        self.ax.plot(
            self.position_list[0, ii + 1],
            self.position_list[1, ii + 1],
            "o",
            color="#135e08",
            markersize=12,
        )
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        plot_obstacles(
            self.ax,
            self.real_obstacle_environment,
            self.x_lim,
            self.y_lim,
            showLabel=False
        )

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            markersize=8,
        )
        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

    def has_converged(self, ii) -> bool:
        return np.allclose(self.position_list[:, ii], self.position_list[:, ii - 1])
