import numpy as np
from math import pi

import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation

from autonomous_furniture.agent import Furniture, Person

import argparse

def test_3D():
    
    #create_table_legs
    table_position = [2,2]
    table_orientation = pi/6
    table_pose = ObjectPose(position=table_position, orientation=table_orientation)
    displacement = [1,1]
    rotation = pi/2
    table_goal = ObjectPose(position=table_position+displacement, orientation=table_orientation+rotation)
    for i in range(4):
        if i == 0:
            leg_local_position = [0.5, 1]
        elif i == 1:
            leg_local_position = [-0.5, 1]
        elif i == 2:
            leg_local_position = [-0.5, -1]
        elif i == 3:
            leg_local_position = [0.5, 1]

        leg_local_orientation = 0.0
        leg_local_pose = ObjectPose(position=np.array(leg_local_position), orientation=leg_local_orientation)
        leg_axes = [0.01, 0.01]
        leg_shape = CuboidXd(
                axes_length = leg_axes,
                pose = leg_local_pose.transform_pose_to_relative(table_pose),
                margin_absolut = 0.01,
                center_position = [0,0],
                )
        obstacle_container = ObstacleContainer()
        leg_ctrpt = np.array([[0,0], [0,0]])
        leg_agent = Furniture(
            shape = leg_shape,
            obstacle_environment = obstacle_container,
            control_points = leg_ctrpt,
            goal_pose = leg_local_pose.transform_pose_to_relative(table_goal)
        )

if (__name__) == "__main__":
    test_3D()