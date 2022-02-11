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
        relative_attractor_position=None,
        goals=None,
        walls=False,
        x_lim=None,
        y_lim=None,
    ):
        num_obs = len(obstacle_environment)
        if start_position.ndim > 1:
            num_agent = len(start_position)
        else:
            num_agent = 1
        dim = 2

        if y_lim is None:
            y_lim = [-3., 3.]
        if x_lim is None:
            x_lim = [-3., 3.]
        if start_position is None:
            start_position = np.zeros((num_obs, self.dim))
        if num_agent > 1:
            velocity = np.zeros((num_agent, self.dim))
        else:
            velocity = np.zeros((2, self.dim))
        if relative_attractor_position is None:
            relative_attractor_position = np.array([0., 0.])
        if goals is None:
            goals = ObstacleContainer()
            goals.append(Cuboid(
                axes_length=[0.6, 0.6],
                center_position=np.array([0., 0.]),
                margin_absolut=0,
                orientation=0,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0., 0.]),
            ))
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
            obstacle_environment = obstacle_environment[:-1] + walls_cont + obstacle_environment[-1:]
        else:
            wall_margin = 0.

        max_axis = max(goals[0].axes_length)
        min_axis = min(goals[0].axes_length)
        offset = 0.5
        wall_thickness = 0.3
        parking_zone_cp = np.array([[x_lim[0] + (wall_margin + wall_thickness), y_lim[1] - (offset + max_axis / 2)],
                                    [x_lim[0] + (wall_margin + wall_thickness), y_lim[0] + (offset + max_axis / 2)],
                                    [x_lim[1] - (wall_margin + wall_thickness), y_lim[1] - (offset + max_axis / 2)],
                                    [x_lim[1] - (wall_margin + wall_thickness), y_lim[0] + (offset + max_axis / 2)]])
        parking_zone_cp = np.array([[x_lim[1] - (wall_margin + wall_thickness), y_lim[0] + (offset + max_axis / 2)],
                                    [x_lim[1] - (wall_margin + wall_thickness + min_axis + offset), y_lim[0] + (offset + max_axis / 2)],
                                    [x_lim[1] - (wall_margin + wall_thickness + 2 * (min_axis + offset)), y_lim[0] + (offset + max_axis / 2)],
                                    [x_lim[1] - (wall_margin + wall_thickness + 3 * (min_axis + offset)), y_lim[0] + (offset + max_axis / 2)]])
        parking_zone = ObstacleContainer()
        for pk in range(len(parking_zone_cp)):
            parking_zone.append(
                Cuboid(
                    axes_length=goals[pk].axes_length,
                    center_position=parking_zone_cp[pk],
                    margin_absolut=0,
                    orientation=pi / 2,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                )
            )

        self.attractor_dynamic = AttractorDynamics(obstacle_environment, cutoff_dist=2, parking_zone=parking_zone)
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
        self.relative_attractor_position = relative_attractor_position
        self.goals = goals
        self.velocity = velocity
        self.parking_zone = parking_zone
        self.num_obs = num_obs
        self.num_agent = num_agent

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment
        self.initial_dynamics = initial_dynamics

        # self.dynamic_avoider = DynamicModulationAvoider(
        #     initial_dynamics=self.initial_dynamics,
        #     environment=self.obstacle_environment,
        # )

        # self.position_list = np.zeros((self.dim, self.it_max))
        # self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        weights = self.dynamic_avoider.get_influence_weight_at_ctl_points(self.position_list[:, :, ii - 1], 3)

        for jj, goal in enumerate(self.goals):
            att_start_time = time.time()
            num_attractor = len(self.obs_w_multi_agent[jj])
            global_attractor_pos = relative2global(self.relative_attractor_position, goal)
            attractor_vel = np.zeros((num_attractor, self.dim))
            for attractor in range(num_attractor):
                attractor_vel[attractor, :], state = self.attractor_dynamic.evaluate(global_attractor_pos[attractor, :], jj)
            attractor_weights = self.attractor_dynamic.get_weights_attractors(global_attractor_pos, jj)
            goal_vel, goal_rot = self.attractor_dynamic.get_goal_velocity(global_attractor_pos, attractor_vel,
                                                                          attractor_weights, jj)
            if state[jj] is False:
                new_goal_pos = goal_vel * self.dt_simulation + goal.center_position
                new_goal_ori = -(1 * goal_rot * self.dt_simulation) + goal.orientation
            else:
                new_goal_pos = self.parking_zone[3-jj].center_position
                new_goal_ori = self.parking_zone[jj].orientation
            goal.center_position = new_goal_pos
            goal.orientation = new_goal_ori

            global_attractor_pos = relative2global(self.relative_attractor_position, goal)
            for i in self.obs_w_multi_agent[jj]:
                self.dynamic_avoider.set_attractor_position(global_attractor_pos[i - (jj * 2)], i)
            att_stop_time = time.time()
            att_tot_time = att_stop_time - att_start_time
            # print(f"Current time of attractor moving: {att_tot_time} for goal: {jj}")

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
                obs_vel = np.array([-0.3, 0.])

            angular_vel = np.zeros(num_agents_in_obs)
            for agent in self.obs_w_multi_agent[obs]:
                angular_vel[agent - (obs * 2)] = weights[obs][agent - (obs * 2)] * np.cross(
                    (self.obstacle_environment[obs].center_position - self.position_list[agent, :, ii - 1]),
                    (self.velocity[agent, :] - obs_vel))

            angular_vel_obs = angular_vel.sum()
            if obs + 1 != self.num_obs:
                self.obstacle_environment[obs].linear_velocity = obs_vel
                self.obstacle_environment[obs].angular_velocity = -2 * angular_vel_obs
                self.obstacle_environment[obs].do_velocity_step(self.dt_simulation)
            else:
                self.obstacle_environment[-1].do_velocity_step(self.dt_simulation)
            for agent in self.obs_w_multi_agent[obs]:
                self.position_list[agent, :, ii] = self.obstacle_environment[obs].transform_relative2global(
                    self.relative_agent_pos[agent, :])

            stop_time = time.time()
            self.time_list[obs, ii - 1] = stop_time - start_time

            # print(f"Max time: {max(time_list[obs, :])}, mean time: {sum(time_list[obs, :])/ii}, for obs: {obs}, with {len(obs_w_multi_agent[obs])} control points")

        # Here come the main calculation part
        # velocity = self.dynamic_avoider.evaluate(self.position_list[:, ii - 1])
        # self.position_list[:, ii] = (
        #     velocity * self.dt_simulation + self.position_list[:, ii - 1]
        # )
        # print(

        # Update obstacles
        # self.obstacle_environment.do_velocity_step(delta_time=self.dt_simulation)

        self.ax.clear()

        # Drawing and adjusting of the axis
        for agent in range(self.num_agent):
            self.ax.plot(
                self.position_list[agent, 0, :ii], self.position_list[agent, 1, :ii], ":", color="#135e08"
            )
            self.ax.plot(
                self.position_list[agent, 0, ii],
                self.position_list[agent, 1, ii],
                "o",
                color="#135e08",
                markersize=12,
            )

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

    def has_converged(self, ii) -> bool:
        # return np.allclose(self.position_list[:, ii], self.position_list[:, ii - 1])
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
    tot_ctl_pts = 8
    obstacle_pos = np.array([[-1.5, 1.5], [-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [4.5, -1.2]])

    rel_agent_pos, radius = calculate_relative_position(num_agent, max_ax_len, min_ax_len)

    obstacle_environment = ObstacleContainer()
    for i in range(len(obstacle_pos) - 1):
        obstacle_environment.append(
            Cuboid(
                axes_length=[max_ax_len, min_ax_len],
                center_position=obstacle_pos[i],
                margin_absolut=radius / 1.1,
                orientation=pi / 2,
                tail_effect=False,
                repulsion_coeff=1,
            )
        )
    obstacle_environment.append(
        Ellipse(
            axes_length=[0.6, 0.6],
            center_position=obstacle_pos[-1],
            margin_absolut=radius,
            orientation=0,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([-0.3, 0.1]),
        )
    )

    agent_pos = np.zeros((tot_ctl_pts, 2))
    for i in range(len(obstacle_pos) - 1):
        agent_pos[(i * 2):(i * 2) + 2] = relative2global(rel_agent_pos, obstacle_environment[i])

    attractor_env = ObstacleContainer()
    for i in range(len(obstacle_pos) - 1):
        attractor_env.append(
            Cuboid(
                axes_length=[max_ax_len, min_ax_len],
                center_position=obstacle_pos[i],
                margin_absolut=0.,
                orientation=pi / 2,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0., 0.]),
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

    obs_multi_agent = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [], 5: []}

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
        it_max=900,
        dt_simulation=0.05,
        dt_sleep=0.01,
        animation_name="full_env_rec",
    )

    my_animation.setup(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        rel_agent_pos,
        attractor_env,
        True,
        x_lim=[-6, 6],
        y_lim=[-5, 5],
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
