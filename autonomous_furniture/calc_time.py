import math
from math import cos, sin
import time
import numpy as np
import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from autonomous_furniture.attractor_dynamics import AttractorDynamics
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num_ctl", action="store", default=2, help="int of number of control points in the furniture")
parser.add_argument("--rect_size", action="store", default="1.5,0.6", help="x,y of the max size of the furniture")
args = parser.parse_args()


class DynamicalSystemAnimation:
    def __init__(self):
        self.animation_paused = True

    def on_click(self, event):
        if self.animation_paused:
            self.animation_paused = False
        else:
            self.animation_paused = True

    def run(
            self, initial_dynamics, obstacle_environment,
            obs_w_multi_agent,
            start_position=None,
            x_lim=None, y_lim=None,
            it_max=1000, dt_step=0.03, dt_sleep=0.1
    ):

        num_obs = len(obstacle_environment)
        if start_position.ndim > 1:
            num_agent = len(start_position)
        else:
            num_agent = 1
        dim = 2

        if y_lim is None:
            y_lim = [-0.5, 2.5]
        if x_lim is None:
            x_lim = [-1.5, 2]
        if start_position is None:
            start_position = np.zeros((num_obs, dim))
        if num_agent > 1:
            velocity = np.zeros((num_agent, dim))
        else:
            velocity = np.zeros((2, dim))

        dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=initial_dynamics, environment=obstacle_environment, obs_multi_agent=obs_w_multi_agent)
        position_list = np.zeros((num_agent, dim, it_max))
        time_list = np.zeros((num_obs, it_max))
        relative_agent_pos = np.zeros((num_agent, dim))

        for obs in range(num_obs):
            for agent in obs_w_multi_agent[obs]:
                if start_position.ndim > 1:
                    relative_agent_pos[agent, :] = - (obstacle_environment[obs].center_position - start_position[agent, :])
                else:
                    relative_agent_pos = - (obstacle_environment[obs].center_position - start_position)

        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots(figsize=(10, 8))  # figsize=(10, 8)
        ax.set_aspect(1.0)
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

        print(f"init dyn: {initial_dynamics[0].attractor_position}")
        print(f"Class dyn avoider: {dynamic_avoider}")

        ii = 0
        while ii < it_max:
            if self.animation_paused:
                plt.pause(dt_sleep)
                if not plt.fignum_exists(fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                continue

            ii += 1
            if ii > it_max:
                break

            # Here come the main calculation part
            weights = dynamic_avoider.get_influence_weight_at_ctl_points(position_list[:, :, ii-1])
            print(f"weights: {weights}")

            if ii % 10 == 0 and ii <= 100:
                # TODO: find a way to move the attractor pos
                pass

            for obs in range(num_obs):
                start_time = time.time()
                num_agents_in_obs = len(obs_w_multi_agent[obs])
                # weights = 1 / len(obs_w_multi_agent)
                for agent in obs_w_multi_agent[obs]:
                    # temp_env = obstacle_environment[0:obs] + obstacle_environment[obs + 1:]
                    temp_env = dynamic_avoider.env_slicer(obs)
                    if (ii % 10) == 0 and ii <= 100:
                        attractor_pos = dynamic_avoider.get_attractor_position(agent)
                        dynamic_avoider.set_attractor_position(attractor_pos+np.array([0.0, 0.05]), agent)
                    velocity[agent, :] = dynamic_avoider.evaluate_for_crowd_agent(position_list[agent, :, ii - 1],
                                                                                  agent, temp_env)
                    velocity[agent, :] = velocity[agent, :] * weights[obs][agent]

                obs_vel = np.zeros(2)
                if obs_w_multi_agent[obs]:
                    for agent in obs_w_multi_agent[obs]:
                        obs_vel += weights[obs][agent] * velocity[agent, :]
                else:
                    obs_vel = np.array([-0.3, 0.0])

                angular_vel = np.zeros(num_agents_in_obs)
                for agent in obs_w_multi_agent[obs]:
                    angular_vel[agent] = weights[obs][agent] * np.cross(
                        (obstacle_environment[obs].center_position - position_list[agent, :, ii - 1]),
                        (velocity[agent, :] - obs_vel))

                angular_vel_obs = angular_vel.sum()
                obstacle_environment[obs].linear_velocity = obs_vel
                obstacle_environment[obs].angular_velocity = -angular_vel_obs
                obstacle_environment[obs].do_velocity_step(dt_step)
                for agent in obs_w_multi_agent[obs]:
                    position_list[agent, :, ii] = obstacle_environment[obs].transform_relative2global(
                        relative_agent_pos[agent, :])

                stop_time = time.time()
                time_list[obs, ii-1] = stop_time-start_time

            print(f"Max time: {max(time_list[0, :])}, mean time: {sum(time_list[0, :])/ii}, for obs: {0}, with {len(obs_w_multi_agent[0])} control points")

            # Clear right before drawing again
            ax.clear()

            # Drawing and adjusting of the axis
            for agent in range(num_agent):
                plt.plot(position_list[agent, 0, :ii], position_list[agent, 1, :ii], ':',
                         color='#135e08')
                plt.plot(position_list[agent, 0, ii], position_list[agent, 1, ii],
                         'o', color='#135e08', markersize=12, )
                plt.arrow(position_list[agent, 0, ii], position_list[agent, 1, ii], velocity[agent, 0],
                          velocity[agent, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            for agent in range(num_agent):
                ax.plot(initial_dynamics[agent].attractor_position[0],
                        initial_dynamics[agent].attractor_position[1], 'k*', markersize=8, )
            ax.grid()

            ax.set_aspect('equal', adjustable='box')
            # breakpoiont()

            # Check convergence
            if np.sum(np.abs(velocity)) < 1e-2:
                print(f"Converged at it={ii}")
                break

            plt.pause(dt_sleep)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break


def calculate_relative_position(num_agent, max_ax, min_ax):
    div = max_ax / (num_agent + 1)
    radius = math.sqrt(((min_ax / 2) ** 2) + (div ** 2))
    rel_agent_pos = np.zeros((num_agent, 2))

    for i in range(num_agent):
        rel_agent_pos[i, 0] = (div * (i + 1)) - (max_ax / 2)

    return rel_agent_pos, radius


def relative2global(relative_pos, obstacle):
    angle = obstacle.orientation
    obs_pos = obstacle.center_position
    print(f"obs pos: {obs_pos}")
    global_pos = np.zeros_like(relative_pos)
    print(f"rel: {relative_pos}")
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    for i in range(relative_pos.shape[0]):
        rot_rel_pos = np.dot(rot, relative_pos[i, :])
        global_pos[i, :] = obs_pos + rot_rel_pos

    print(f"glob: {global_pos}")
    return global_pos


def multiple_robots():
    center_point = 3.0
    num_agent = int(args.num_ctl)
    str_axis = args.rect_size.split(",")
    axis = [float(str_axis[0]), float(str_axis[1])]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    # div = max_ax_len / (num_agent + 1)
    # radius = math.sqrt(((min_ax_len / 2) ** 2) + (div ** 2))
    obstacle_pos = np.array([[-center_point, 0.0], [3.0, -0.5]])
    # agent_pos = np.zeros((num_agent, 2))
    # for i in range(num_agent):
    #     agent_pos[i, 0] = - center_point + ((div * (i+1)) - (max_ax_len / 2))
    attractor_pos = np.zeros((num_agent, 2))
    # for i in range(num_agent):
    #     attractor_pos[i, 0] = 1.0
    #     attractor_pos[i, 1] = (div * (i+1)) - (max_ax_len / 2)

    rel_agent_pos, radius = calculate_relative_position(num_agent, max_ax_len, min_ax_len)

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(Cuboid(
        axes_length=[max_ax_len, min_ax_len],
        center_position=obstacle_pos[0],
        margin_absolut=0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
    ))
    obstacle_environment.append(Ellipse(
        axes_length=[0.6, 0.6],
        center_position=obstacle_pos[1],
        margin_absolut=radius,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([-0.3, 0.0]),
    ))

    agent_pos = relative2global(rel_agent_pos, obstacle_environment[0])

    attractor_env = ObstacleContainer()
    attractor_env.append(Ellipse(
        axes_length=[0.6, 0.6],
        center_position=np.array([1., 0.]),
        margin_absolut=radius,
        orientation=math.pi/2,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([0., 0.]),
    ))

    attractor_pos = relative2global(rel_agent_pos, attractor_env[0])

    initial_dynamics = [LinearSystem(
        attractor_position=attractor_pos[0],
        maximum_velocity=1, distance_decrease=0.3
    ),
        LinearSystem(
            attractor_position=attractor_pos[1],
            maximum_velocity=1, distance_decrease=0.3
        )
    ]
    initial_dynamics = []
    for i in range(num_agent):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1, distance_decrease=0.3
            )
        )

    obs_multi_agent = {0: [0, 1, 2], 1: []}
    obs_multi_agent = {0: [], 1: []}
    for i in range(num_agent):
        obs_multi_agent[0].append(i)

    DynamicalSystemAnimation().run(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        x_lim=[-4, 3],
        y_lim=[-3, 3],
        dt_step=0.05,
    )


if __name__ == "__main__":
    plt.close('all')
    plt.ion()

    multiple_robots()
