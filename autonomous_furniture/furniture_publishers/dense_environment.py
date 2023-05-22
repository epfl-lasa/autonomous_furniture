from typing import Optional
import math
import copy

import numpy as np
import matplotlib.pyplot as plt

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from autonomous_furniture.agent import ObjectType
from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.agent import Furniture, Person

from autonomous_furniture.obstacle_container import GlobalObstacleContainer
from autonomous_furniture.furniture_creators import create_four_chair_arrangement
from autonomous_furniture.furniture_creators import (
    set_goals_to_arrange_on_the_left,
    set_goals_to_arrange_on_the_right,
)


def create_environment(go_to_center: bool = False):
    x_lim: list[float] = [-6.0, 6.0]
    y_lim: list[float] = [-5.0, 5.0]

    GlobalObstacleContainer().empty()

    agent_list: list[BaseAgent] = []
    agent_list = agent_list + create_four_chair_arrangement(np.array([-2, -2.0]))
    agent_list = agent_list + create_four_chair_arrangement(np.array([2.0, -2.0]))
    agent_list = agent_list + create_four_chair_arrangement(np.array([-2.0, 2.0]))
    agent_list = agent_list + create_four_chair_arrangement(np.array([2.0, 2.0]))

    set_goals_to_arrange_on_the_right(
        agent_list,
        x_lim=x_lim,
        y_lim=y_lim,
        delta_x=1.2,
        delta_y=3.0,
        orientation=math.pi / 2.0,
        arranging_type=ObjectType.TABLE,
    )

    set_goals_to_arrange_on_the_left(
        agent_list,
        x_lim=x_lim,
        y_lim=y_lim,
        delta_x=1.0,
        delta_y=1.2,
        orientation=math.pi / 2.0,
        # orientation=0.0,
        arranging_type=ObjectType.CHAIR,
    )

    if go_to_center:
        for agent in agent_list:
            tmp_goal = copy.deepcopy(agent._goal_pose)
            agent.set_goal_pose(copy.deepcopy(agent.pose))
            agent._shape.pose = tmp_goal

    return agent_list, x_lim, y_lim


def run_chair_and_table_matplotlib(go_to_center: bool = False):
    """Matplotlib Animation with many chairs and tables (!)."""
    import matplotlib.pyplot as plt
    from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation

    plt.ion()

    agent_list, x_lim, y_lim = create_environment(go_to_center=go_to_center)

    my_animation = DynamicalSystemAnimation(
        it_max=800,
        dt_simulation=0.02,
        dt_sleep=0.05,
        animation_name="dense_furniture_animation",
        # file_type=".mp4",
        file_type=".gif",
    )

    my_animation.setup(
        obstacle_environment=GlobalObstacleContainer(),
        agent=agent_list,
        x_lim=x_lim,
        y_lim=y_lim,
        figsize=(5, 4),
        version="v2",
        mini_drag="dragdist",
        check_convergence=False,
        obstacle_colors=[],
    )

    my_animation.run(save_animation=False)
    my_animation.logs(len(agent_list))
    print("Done")


def run_chair_and_table_rviz(go_to_center: bool = False):
    """Run environment using ros and rviz"""
    import rclpy
    from rclpy.node import Node
    from autonomous_furniture.rviz_animator import RvizSimulator

    agent_list, x_lim, y_lim = create_environment(go_to_center)

    ## Start ROS Node
    print("Starting publishing node")
    rclpy.init()
    visualizer = RvizSimulator(
        it_max=800,
        dt_simulation=0.02,
        dt_sleep=0.05,
    )
    visualizer.setup(
        obstacle_environment=GlobalObstacleContainer(),
        agent=agent_list,
    )

    rclpy.spin(visualizer)
    try:
        rckpy.shutdown()
    except:
        breakpoint()


if (__name__) == "__main__":
    plt.close("all")
    # main()
    run_chair_and_table_matplotlib(go_to_center=True)
    # run_chair_and_table_rviz(go_to_center=True)

    print("Done")
