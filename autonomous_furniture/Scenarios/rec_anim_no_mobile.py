from math import pi, cos, sin, sqrt

import numpy as np

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from autonomous_furniture.attractor_dynamics import AttractorDynamics

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
args = parser.parse_args()


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


def calculate_relative_position(num_agent, max_ax, min_ax):
    div = max_ax / (num_agent + 1)
    radius = sqrt(((min_ax / 2) ** 2) + (div ** 2))
    rel_agent_pos = np.zeros((num_agent, 2))

    for i in range(num_agent):
        rel_agent_pos[i, 0] = (div * (i + 1)) - (max_ax / 2)

    return rel_agent_pos, radius


def relative2global(relative_pos, obstacle):
    angle = obstacle.orientation
    obs_pos = obstacle.center_position
    global_pos = np.zeros_like(relative_pos)
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    for i in range(relative_pos.shape[0]):
        rot_rel_pos = np.dot(rot, relative_pos[i, :])
        global_pos[i, :] = obs_pos + rot_rel_pos

    return global_pos


def global2relative(global_pos, obstacle):
    angle = -1 * obstacle.orientation
    obs_pos = obstacle.center_position
    relative_pos = np.zeros_like(global_pos)
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    for i in range(global_pos.shape[0]):
        rel_pos_pre_rot = global_pos[i, :] - obs_pos
        relative_pos[i, :] = np.dot(rot, rel_pos_pre_rot)

    return relative_pos


def run_person_avoiding_multiple_furniture():
    axis = [2.2, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    tot_ctl_pts = 2
    attractor_cp_pos = np.array([[-5, 2], [0, 1]])

    obstacle_pos = np.array([[-1.5, 1.5], [-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [4.5, -1.2]])

    radius = 0.6

    obstacle_environment = ObstacleContainer()
    for i in range(len(obstacle_pos) - 1):
        obstacle_environment.append(
            Cuboid(
                axes_length=[max_ax_len, min_ax_len],
                center_position=obstacle_pos[i],
                margin_absolut=radius,
                orientation=pi / 2,
                tail_effect=False,
                repulsion_coeff=1,
            )
        )
    obstacle_environment.append(
        Ellipse(
            axes_length=[0.6, 0.6],
            center_position=obstacle_pos[-1],
            margin_absolut=0,
            orientation=0,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([-0.3, 0.1]),
        )
    )

    agent_pos = np.zeros((tot_ctl_pts, 2))
    agent_pos[0] = obstacle_pos[-1]

    attractor_env = ObstacleContainer()
    attractor_env.append(
        Ellipse(
            axes_length=[0.6, 0.6],
            center_position=attractor_cp_pos[0],
            margin_absolut=0,
            orientation=0,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([0., 0.]),
        )
    )

    attractor_pos = np.zeros((tot_ctl_pts, 2))
    attractor_pos[0] = attractor_cp_pos[0]

    initial_dynamics = LinearSystem(
        attractor_position=attractor_pos[0],
        maximum_velocity=1,
        distance_decrease=0.3
    )

    obs_multi_agent = {4: [0]}

    my_animation = DynamicalSystemAnimation(
        it_max=450,
        dt_simulation=0.05,
        dt_sleep=0.01,
        animation_name="no_smart_furniture",
    )

    my_animation.setup(
        initial_dynamics,
        obstacle_environment,
        agent_pos[0],
        False,
        x_lim=[-6, 6],
        y_lim=[-5, 5],
    )

    my_animation.run(save_animation=args.rec)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    run_person_avoiding_multiple_furniture()
