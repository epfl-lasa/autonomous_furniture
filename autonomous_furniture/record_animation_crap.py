import time
import os
import datetime
from math import pi, cos, sin, sqrt

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from autonomous_furniture.attractor_dynamics import AttractorDynamics


class DynamicalSystemAnimation(Animator):
    dim = 2

    def setup(
        self,
        initial_dynamics,
        obstacle_environment,
        obs_w_multi_agent,
        start_position=None,
        x_lim=None,
        y_lim=None,
    ):
        num_obs = len(obstacle_environment)
        num_agent = len(start_position)
        dim = 2

        if y_lim is None:
            y_lim = [-3., 3.]
        if x_lim is None:
            x_lim = [-3., 3.]
        if start_position is None:
            start_position = np.zeros((num_obs, self.dim))
        if num_agent > 1:
            velocity = np.zeros((num_agent, self.dim))

        self.dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=initial_dynamics, environment=obstacle_environment,
                                                   obs_multi_agent=obs_w_multi_agent)
        self.position_list = np.zeros((num_agent, dim, self.it_max))
        self.time_list = np.zeros((num_obs, self.it_max))
        self.relative_agent_pos = np.zeros((num_agent, dim))

        for obs in range(num_obs):
            self.relative_agent_pos[obs_w_multi_agent[obs], :] = global2relative(start_position[obs_w_multi_agent[obs]],
                                                                                 obstacle_environment[obs])

        self.position_list[:, :, 0] = start_position

        self.obs_w_multi_agent = obs_w_multi_agent
        self.velocity = velocity
        self.num_obs = num_obs
        self.num_agent = num_agent

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment
        self.initial_dynamics = initial_dynamics

        self.color_list = ["#4472c4", "#ed7d31"]
        self.agent_weights = np.zeros((self.it_max, num_agent))

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        weights = self.dynamic_avoider.get_influence_weight_at_ctl_points(self.position_list[:, :, ii - 1], 3)
        self.agent_weights[ii, :] = weights[0]

        for obs in range(self.num_obs):
            start_time = time.time()
            num_agents_in_obs = len(self.obs_w_multi_agent[obs])
            # weights = 1 / len(obs_w_multi_agent)
            for agent in self.obs_w_multi_agent[obs]:
                temp_env = self.dynamic_avoider.env_slicer(obs)
                self.velocity[agent, :] = self.dynamic_avoider.evaluate_for_crowd_agent(self.position_list[agent, :, ii - 1],
                                                                                        agent, temp_env)
                self.velocity[agent, :] = self.velocity[agent, :] * weights[obs][agent - (obs * 2)]

            obs_vel = np.zeros(2)
            if self.obs_w_multi_agent[obs]:
                for agent in self.obs_w_multi_agent[obs]:
                    obs_vel += weights[obs][agent - (obs * 2)] * self.velocity[agent, :]
            else:
                obs_vel = np.array([0., 0.])

            angular_vel = np.zeros(num_agents_in_obs)
            for agent in self.obs_w_multi_agent[obs]:
                angular_vel[agent - (obs * 2)] = weights[obs][agent - (obs * 2)] * np.cross(
                    (self.obstacle_environment[obs].center_position - self.position_list[agent, :, ii - 1]),
                    (self.velocity[agent, :] - obs_vel))

            angular_vel_obs = angular_vel.sum()
            self.obstacle_environment[obs].linear_velocity = obs_vel
            self.obstacle_environment[obs].angular_velocity = -2 * angular_vel_obs
            self.obstacle_environment[obs].do_velocity_step(self.dt_simulation)
            for agent in self.obs_w_multi_agent[obs]:
                self.position_list[agent, :, ii] = self.obstacle_environment[obs].transform_relative2global(
                    self.relative_agent_pos[agent, :])

            stop_time = time.time()
            self.time_list[obs, ii - 1] = stop_time - start_time

        # Update obstacles

        self.ax.clear()

        # Drawing and adjusting of the axis
        for agent in range(self.num_agent):
            self.ax.plot(
                self.position_list[agent, 0, :ii],
                self.position_list[agent, 1, :ii],
                ":",
                color="#135e08"
            )
            self.ax.plot(
                self.position_list[agent, 0, ii],
                self.position_list[agent, 1, ii],
                "o",
                color=self.color_list[agent],
                markersize=12 * 2 * weights[0][agent],
            )
            self.ax.arrow(self.position_list[agent, 0, ii],
                          self.position_list[agent, 1, ii],
                          self.velocity[agent, 0],
                          self.velocity[agent, 1],
                          head_width=0.05,
                          head_length=0.1,
                          fc='k',
                          ec='k')

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        plot_obstacles(
            self.ax, self.obstacle_environment, self.x_lim, self.y_lim, showLabel=False
        )

        for agent in range(self.num_agent):
            self.ax.plot(
                self.initial_dynamics[agent].attractor_position[0],
                self.initial_dynamics[agent].attractor_position[1],
                "k*",
                markersize=8,
            )
        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

        if ii == self.it_max - 1:
            source = "data/"
            temp_array = np.asarray(self.agent_weights)
            np.savetxt(source + "rot_agent_weights" + ".csv", temp_array, delimiter=",")

    def has_converged(self, ii) -> bool:
        # return np.allclose(self.position_list[0, :, ii], self.position_list[0, :, ii - 1])
        return False


def simple_point_robot():
    """Simple robot avoidance."""
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[0.6, 1.3],
            center_position=np.array([-0.2, 2.4]),
            margin_absolut=0,
            orientation=-30 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 1.3],
            center_position=np.array([1.2, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.5,
            orientation=10 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.0, 1.8]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )


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
    # print(f"obs pos: {obs_pos}")
    global_pos = np.zeros_like(relative_pos)
    # print(f"rel: {relative_pos}")
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    for i in range(relative_pos.shape[0]):
        rot_rel_pos = np.dot(rot, relative_pos[i, :])
        global_pos[i, :] = obs_pos + rot_rel_pos

    # print(f"glob: {global_pos}")
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


def run_multiple_furniture_avoiding_person():
    num_agent = 2
    axis = [2.2, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    tot_ctl_pts = 2
    obstacle_pos = np.array([[-2., 0.], [2., -2.]])

    rel_agent_pos, radius = calculate_relative_position(num_agent, max_ax_len, min_ax_len)

    attractor_env_pos = np.array([[1.0, 0.]])

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[0],
            margin_absolut=0.,
            orientation=0.,
            tail_effect=False,
            repulsion_coeff=1,
        )
    )
    obstacle_environment.append(
        Ellipse(
            axes_length=[.5, .5],
            center_position=obstacle_pos[1],
            margin_absolut=radius,
            orientation=0.,
            tail_effect=False,
            repulsion_coeff=1,
        )
    )

    agent_pos = np.zeros((tot_ctl_pts, 2))
    for i in range(len(obstacle_pos) - 1):
        agent_pos[(i * 2):(i * 2) + 2] = relative2global(rel_agent_pos, obstacle_environment[i])

    attractor_env = ObstacleContainer()
    attractor_env.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=attractor_env_pos[0],
            margin_absolut=0.,
            orientation=pi / 2,
            tail_effect=False,
            repulsion_coeff=1,
        )
    )

    attractor_pos = np.zeros((tot_ctl_pts, 2))
    for i in range(len(obstacle_pos) - 1):
        attractor_pos[(i * 2):(i * 2) + 2] = relative2global(rel_agent_pos, attractor_env[i])

    initial_dynamics = []
    for i in range(tot_ctl_pts):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1, distance_decrease=0.3
            )
        )

    obs_multi_agent = {0: [0, 1], 1: []}

    # replace this with
    # DynamicalSystemAnimation().run(
    #     initial_dynamics,
    #     obstacle_environment,
    #     obs_multi_agent,
    #     agent_pos,
    #     rel_agent_pos,
    #     attractor_env,
    #     True,
    #     x_lim=[-6, 6],
    #     y_lim=[-5, 5],
    #     dt_step=0.03,
    #     dt_sleep=0.01,
    # )

    # this
    my_animation = DynamicalSystemAnimation(
        it_max=450,
        dt_simulation=0.05,
        dt_sleep=0.01,
        animation_name="rotating_agent_color",
    )

    my_animation.setup(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        x_lim=[-4, 4],
        y_lim=[-3, 3],
    )

    # code from Lukas
    # obstacle_environment = ObstacleContainer()
    # obstacle_environment.append(
    #     Ellipse(
    #         axes_length=[0.5, 0.5],
    #         # center_position=np.array([-3.0, 0.2]),
    #         center_position=np.array([-1.0, 0.2]),
    #         margin_absolut=0.5,
    #         orientation=0,
    #         linear_velocity=np.array([0.5, 0.0]),
    #         tail_effect=False,
    #     )
    # )
    #
    # initial_dynamics = LinearSystem(
    #     attractor_position=np.array([0.0, 0.0]),
    #     maximum_velocity=1,
    #     distance_decrease=0.3,
    # )
    #
    # my_animation = DynamicalSystemAnimation(
    #     dt_simulation=0.05,
    #     dt_sleep=0.01,
    # )
    #
    # my_animation.setup(
    #     initial_dynamics,
    #     obstacle_environment,
    #     x_lim=[-3, 3],
    #     y_lim=[-2.1, 2.1],
    # )

    my_animation.run(save_animation=True)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    # simple_point_robot()
    run_multiple_furniture_avoiding_person()
