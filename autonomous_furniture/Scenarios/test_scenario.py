#!/USSR/bin/python3
"""

"""
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


class DynamicalSystemAnimation:
    def __init__(self):
        self.animation_paused = False

    def on_click(self, event):
        if self.animation_paused:
            self.animation_paused = False
        else:
            self.animation_paused = True

    def run(
            self, initial_dynamics, obstacle_environment,
            start_position=np.array([[0, 0.01], [0, -1]]),
            x_lim=None, y_lim=None,
            it_max=1000, dt_step=0.03, dt_sleep=0.1
    ):

        if y_lim is None:
            y_lim = [-0.5, 2.5]
        if x_lim is None:
            x_lim = [-1.5, 2]

        dynamic_avoider = DynamicCrowdAvoider(
            initial_dynamics=initial_dynamics, environment=obstacle_environment)

        dim = 2
        num_agent = len(start_position)
        position_list = np.zeros((num_agent, dim, it_max))
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
            for agent in range(num_agent):
                velocity = dynamic_avoider.evaluate_for_crowd_agent(position_list[agent, :, ii - 1], agent, False)
                position_list[agent, :, ii] = velocity * dt_step + position_list[agent, :, ii - 1] #+ np.random.uniform(-0.01, 0.01, 2)

            obstacle_environment[0].center_position += np.array([0.05, 0])
            obstacle_environment[1].center_position -= np.array([0, 0.02])

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


def simple_point_robot():
    """ Simple robot avoidance. """
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(axes_length=[0.5, 0.5],
                center_position=np.array([-2.0, 0.0]),
                margin_absolut=0.4,
                orientation=0,
                tail_effect=False,
                repulsion_coeff=1.4,
                is_dynamic=True,
                linear_velocity=np.array([0.05, 0]),
                )
    )
    obstacle_environment.append(
        Ellipse(axes_length=[0.5, 0.5],
                center_position=np.array([0.0, 2.0]),
                margin_absolut=0.4,
                orientation=0,
                tail_effect=False,
                repulsion_coeff=1.4,
                is_dynamic=True,
                linear_velocity=np.array([0, -0.02]),
                )
    )

    initial_dynamics = [LinearSystem(
        attractor_position=np.array([0, 0]),
        maximum_velocity=1, distance_decrease=0.3,
    ), LinearSystem(
        attractor_position=np.array([0, 0]),
        maximum_velocity=1, distance_decrease=0.3
    )]

    DynamicalSystemAnimation().run(
        initial_dynamics, obstacle_environment,
        x_lim=[-3, 3], y_lim=[-3, 3],
        dt_step=0.05,
    )


if __name__ == "__main__":
    plt.close('all')
    plt.ion()

    simple_point_robot()
