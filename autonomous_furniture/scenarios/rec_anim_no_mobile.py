from math import pi, cos, sin, sqrt

import numpy as np

import matplotlib.pyplot as plt

from animator_class import DynamicalSystemAnimation

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from vartools.dynamical_systems import LinearSystem

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
args = parser.parse_args()


def calculate_relative_position(num_agent, max_ax, min_ax):
    div = max_ax / (num_agent + 1)
    radius = sqrt(((min_ax / 2) ** 2) + (div**2))
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


def energyComputation():
    pass


def run_person_avoiding_multiple_furniture():
    axis = [2.2, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    tot_ctl_pts = 2
    attractor_cp_pos = np.array([[-5, 2], [0, 1]])

    obstacle_pos = np.array(
        [[-1.5, 1.5], [-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [4.5, -1.2]]
    )

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
            linear_velocity=np.array([0.0, 0.0]),
        )
    )

    attractor_pos = np.zeros((tot_ctl_pts, 2))
    attractor_pos[0] = attractor_cp_pos[0]

    initial_dynamics = LinearSystem(
        attractor_position=attractor_pos[0], maximum_velocity=1, distance_decrease=0.3
    )

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
