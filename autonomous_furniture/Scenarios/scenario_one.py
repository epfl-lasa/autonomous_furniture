# Author: Federico Conzelmann
# Email: federico.conzelmann@epfl.ch
# Created: 2021-10-28

from math import pi

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem


class DynamicalSystemScenario:
    def __init__(self):
        self.animation_paused = False

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

        num_agent = len(start_position)
        num_obs = len(obstacle_environment)
        dim = 2
        velocity = np.zeros((num_agent, dim))

        if y_lim is None:
            y_lim = [-0.5, 2.5]
        if x_lim is None:
            x_lim = [-1.5, 2]
        if start_position is None:
            start_position = np.zeros((num_agent, dim))

        dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=initial_dynamics, environment=obstacle_environment)
        position_list = np.zeros((num_agent, dim, it_max))
        relative_agent_pos = np.zeros((num_agent, dim))

        for obs in range(num_obs):
            for agent in obs_w_multi_agent[obs]:
                relative_agent_pos[agent, :] = - (obstacle_environment[obs].center_position - start_position[agent, :])

        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots(figsize=(10, 8))
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

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
            if not ii % 10:
                # print(f"it={ii}")
                pass

            # Here come the main calculation part
            for obs in range(num_obs):
                temp_env = dynamic_avoider.env_slicer(obs)
                for agent in obs_w_multi_agent[obs]:
                    max_axis = max(obstacle_environment[obs].axes_length)
                    obstacle_environment[obs].margin_absolut = max_axis
                    velocity[agent, :] = dynamic_avoider.evaluate_for_crowd_agent(position_list[agent, :, ii - 1], agent, temp_env)
                    obstacle_environment[obs].linear_velocity = velocity[agent, :]
                    # position_list[agent, :, ii] = velocity[agent, :] * dt_step + position_list[agent, :, ii - 1] #+ np.random.uniform(-0.01, 0.01, 2)

                obstacle_environment[obs].do_velocity_step(dt_step)
                for agent in obs_w_multi_agent[obs]:
                    position_list[agent, :, ii] = obstacle_environment[obs].transform_relative2global(
                        relative_agent_pos[agent, :])

            # if i do it like this the agent has to be in the last index of the "position_list"
            # for obstacle in range(num_obs):
            #     if obstacle < num_agent:
            #         obstacle_environment[obstacle].center_position = position_list[obstacle, :, ii]
            #         print("is this point reached ? 1")
            #     else:
            #         obstacle_environment[obstacle].center_position += np.array([0.05, 0])
            #         print("is this point reached ? 2")

            # Clear right before drawing again
            ax.clear()

            # Drawing and adjusting of the axis
            for agent in range(num_agent):
                plt.plot(position_list[agent, 0, :ii], position_list[agent, 1, :ii], ':',
                         color='#135e08')
                plt.plot(position_list[agent, 0, ii], position_list[agent, 1, ii],
                         'o', color='#135e08', markersize=12, )

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
            # if np.sum(np.abs(velocity)) < 1e-2:
            #     print(f"Converged at it={ii}")
            #     break

            plt.pause(dt_sleep)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break


def multiple_robots():
    obstacle_pos = np.array([[0, 0.01], [0.4, 0.61], [-0.41, 0.6], [-0.41, -0.6], [0.4, -0.61], [-2.0, 0.0]])
    agent_pos = obstacle_pos[0:-1]
    attractor_pos = np.array([[0.0, 0.0], [0.4, 0.6], [-0.4, 0.6], [-0.4, -0.6], [0.4, -0.6]])
    obstacle_environment = ObstacleContainer()
    # Table
    obstacle_environment.append(Cuboid(
        axes_length=[1.6, 0.7],
        center_position=obstacle_pos[0],
        margin_absolut=0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        is_dynamic=True,
    ))
    # Chair
    obstacle_environment.append(Cuboid(
        axes_length=[0.5, 0.5],
        center_position=obstacle_pos[1],
        margin_absolut=0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        is_dynamic=True,
    ))
    # Chair 2
    obstacle_environment.append(Cuboid(
        axes_length=[0.5, 0.5],
        center_position=obstacle_pos[2],
        margin_absolut=0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        is_dynamic=True,
    ))
    # Chair 3
    obstacle_environment.append(Cuboid(
        axes_length=[0.5, 0.5],
        center_position=obstacle_pos[3],
        margin_absolut=0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        is_dynamic=True,
    ))
    # Chair 4
    obstacle_environment.append(Cuboid(
        axes_length=[0.5, 0.5],
        center_position=obstacle_pos[4],
        margin_absolut=0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        is_dynamic=True,
    ))
    # Person
    obstacle_environment.append(Ellipse(
        axes_length=[0.5, 0.5],
        center_position=obstacle_pos[5],
        margin_absolut=0.5,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1.4,
        is_dynamic=True,
        linear_velocity=np.array([0.1, 0]),
    ))
    initial_dynamics = [LinearSystem(
        attractor_position=attractor_pos[0],
        maximum_velocity=1, distance_decrease=0.3
    ), LinearSystem(
        attractor_position=attractor_pos[1],
        maximum_velocity=1, distance_decrease=0.3
    ), LinearSystem(
        attractor_position=attractor_pos[2],
        maximum_velocity=1, distance_decrease=0.3
    ), LinearSystem(
        attractor_position=attractor_pos[3],
        maximum_velocity=1, distance_decrease=0.3
    ), LinearSystem(
        attractor_position=attractor_pos[4],
        maximum_velocity=1, distance_decrease=0.3
    )]

    obs_multi_agent = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: []}

    DynamicalSystemScenario().run(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        x_lim=[-3, 3],
        y_lim=[-3, 3],
        dt_step=0.05,
    )


if __name__ == "__main__":
    plt.close('all')
    plt.ion()

    multiple_robots()