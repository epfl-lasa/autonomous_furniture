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


# Compute the relative postion of the Control Point inside the obstacle
def calculate_relative_position(num_agent, max_ax, min_ax):
    div = max_ax / (num_agent + 1) 
    radius = sqrt(((min_ax / 2) ** 2) + (div ** 2))
    rel_agent_pos = np.zeros((num_agent, 2))

    for i in range(num_agent):
        rel_agent_pos[i, 0] = (div * (i + 1)) - (max_ax / 2) # postion of the controle point inside the obstacle

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

def main():
    num_agent = 2       # possibly the number of control point per obstacle (likely only the cuboid one)
    axis = [2.2, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    tot_ctl_pts = 8 # total number of Control point that will have to be treated i.e. num_agent*numb_cuboid_obstacle()
    obstacle_pos = np.array([[-1.5, 1.5], [-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [4.5, -1.2]])

    rel_agent_pos, radius = calculate_relative_position(num_agent, max_ax_len, min_ax_len)

    ## Creation of obstacles in the env : 
    obstacle_environment = ObstacleContainer()
    for i in range(len(obstacle_pos) - 1):
        obstacle_environment.append(
            Cuboid(
                axes_length=[max_ax_len, min_ax_len],
                center_position=obstacle_pos[i],
                margin_absolut=radius / 1.1, # TODO Why do we devide by a factor the margin
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

    agent_pos = np.zeros((tot_ctl_pts, 2)) # List with all the control points off all the obstacle
    for i in range(len(obstacle_pos) - 1):
        agent_pos[(i * 2):(i * 2) + 2] = relative2global(rel_agent_pos, obstacle_environment[i]) #agent_pos[n] -> [a,b] TODO Here could replace 2 \
                                                                                    # by the num_agent i.e. number of control point per obstacle

    attractor_env = ObstacleContainer()
    for i in range(len(obstacle_pos) - 1): # the cuboid obstacles are also added in an attractor env
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

    attractor_pos = np.zeros((tot_ctl_pts, 2)) # and we do similar thing to position the attractor (will be in the same place as the Control point)
    for i in range(len(obstacle_pos) - 1):
        attractor_pos[(i * 2):(i * 2) + 2] = relative2global(rel_agent_pos, attractor_env[i]) #agent_pos[n] -> [a,b] TODO Here could replace 2 \
                                                                                    # by the num_agent i.e. number of control point per obstacle

    initial_dynamics = []
    for i in range(tot_ctl_pts):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1, distance_decrease=0.3
            )
        )
if __name__ == "__main__":
    main()