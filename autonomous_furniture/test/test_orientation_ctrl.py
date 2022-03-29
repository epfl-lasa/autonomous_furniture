from math import pi, cos, sin, sqrt
from turtle import shape

import numpy as np

import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from autonomous_furniture.attractor_dynamics import AttractorDynamics

from autonomous_furniture.agent import Furniture, Person

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
args = parser.parse_args()


class DynamicalSystemAnimation(Animator):
    def setup(
        self,
        obstacle_environment,
        agent,
        x_lim=None,
        y_lim=None,
    ):

        dim = 2
        self.number_agent = len(agent)

        if y_lim is None:
            y_lim = [0., 10]
        if x_lim is None:
            x_lim = [0, 10]

        # self.attractor_dynamic = AttractorDynamics(obstacle_environment, cutoff_dist=1.8, parking_zone=parking_zone)
        # self.dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=initial_dynamics, environment=obstacle_environment,
        #                                           obs_multi_agent=obs_w_multi_agent)
        self.position_list = np.zeros((dim, self.it_max))
        self.time_list = np.zeros((self.it_max))
        self.position_list = [
            agent[ii].position for ii in range(self.number_agent)]
        self.agent = agent
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment

        self.fig, self.ax = plt.subplots()

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        self.ax.clear()

        # Drawing and adjusting of the axis
        # for agent in range(self.num_agent):
        #     self.ax.plot(
        #         self.position_list[agent, 0, :ii + 1],
        #         self.position_list[agent, 1, :ii + 1],
        #         ":",
        #         color="#135e08"
        #     )
        #     self.ax.plot(
        #         self.position_list[agent, 0, ii + 1],
        #         self.position_list[agent, 1, ii + 1],
        #         "o",
        #         color="#135e08",
        #         markersize=12,
        #     )

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        plot_obstacles(
            self.ax, self.obstacle_environment, self.x_lim, self.y_lim, showLabel=False
        )

        for jj in range(self.number_agent):
            self.agent[jj].update_velocity()
            self.agent[jj].do_velocity_step(self.dt_simulation)
            global_crontrol_points = self.agent[jj].get_global_control_points()
            self.ax.plot(
                global_crontrol_points[0, :], global_crontrol_points[1, :], 'ko')

            goal_crontrol_points = self.agent[jj].get_goal_control_points()
            self.ax.plot(
                goal_crontrol_points[0, :], goal_crontrol_points[1, :], 'ko')

        # for agent in range(self.num_agent):
        #     plt.arrow(self.position_list[agent, 0, ii + 1],
        #               self.position_list[agent, 1, ii + 1],
        #               self.velocity[agent, 0],
        #               self.velocity[agent, 1],
        #               head_width=0.05,
        #               head_length=0.1,
        #               fc='k',
        #               ec='k')

        #     self.ax.plot(
        #         self.initial_dynamics[agent].attractor_position[0],
        #         self.initial_dynamics[agent].attractor_position[1],
        #         "k*",
        #         markersize=8,
        #     )
        # self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

    def has_converged(self, ii) -> bool:
        # return np.allclose(self.position_list[:, ii], self.position_list[:, ii - 1])
        return False

def run_single_furniture_rotating():
    axis = [2.2, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    obstacle_environment = ObstacleContainer() # List of environment shared by all the furniture/agent
    
    control_points = np.array([[0.4, 0], [-0.4, 0]]) # control_points for the cuboid

    goal = ObjectPose(position=np.array([6, 3]))  # , orientation = 1.6) Goal of the CuboidXd
    
    table_shape = CuboidXd(axes_length=[max_ax_len, min_ax_len],
                           center_position=np.array([0, 3]),
                           margin_absolut=0.6,
                           orientation=np.pi/2,
                           tail_effect=False,)

    my_furniture = [Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal)]

    my_animation = DynamicalSystemAnimation(
        it_max=450,
        dt_simulation=0.05,
        dt_sleep=0.01,
        animation_name="rotating_agent",
    )

    my_animation.setup(
        obstacle_environment,
        agent=my_furniture,
        x_lim=[-3, 8],
        y_lim=[-2, 7],
    )

    my_animation.run(save_animation=args.rec)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    run_single_furniture_rotating()