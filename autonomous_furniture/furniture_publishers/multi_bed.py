from typing import Optional

import math

import numpy as np

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from autonomous_furniture.agent import ObjectType
from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.agent import Furniture, Person

from autonomous_furniture.obstacle_container import GlobalObstacleContainer
from autonomous_furniture.furniture_creators import create_hospital_bed
from autonomous_furniture.evaluation.scenario_launcher import ScenarioLauncher


def create_environment(n_agents: int = 10):
    import random

    # random.seed(7)
    random.seed(11)

    obstacle_environment = GlobalObstacleContainer()

    sample_bed = create_hospital_bed(
        start_pose=ObjectPose(position=np.zeros(2), orientation=0)
    )
    my_scenario = ScenarioLauncher(
        nb_furniture=n_agents, furniture_shape=sample_bed._shape
    )
    my_scenario.creation()

    obstacle_environment.empty()

    agent_list: list[BaseAgent] = []
    for ii in range(n_agents):
        new_bed = create_hospital_bed(
            start_pose=my_scenario._init_setup[ii],
            goal_pose=my_scenario._goal_setup[ii],
        )
        agent_list.append(new_bed)

    return agent_list


def run_ten_bed_animation_matplotlib(it_max=800):
    # Get simple bed - to input it into the ScenarioLauncher
    import matplotlib.pyplot as plt
    from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation

    agent_list = create_environment()

    n_agents = len(agent_list)
    cm = plt.get_cmap("gist_rainbow")
    color_list = [cm(1.0 * ii / n_agents) for ii in range(n_agents)]

    my_animation = DynamicalSystemAnimation(
        it_max=it_max,
        dt_simulation=0.01,
        dt_sleep=0.2,
        animation_name="multiple_bed_animation",
        file_type=".gif",
    )

    my_animation.setup(
        obstacle_environment=GlobalObstacleContainer(),
        agent=agent_list,
        x_lim=[0, 11],
        y_lim=[0, 9],
        figsize=(5, 4),
        version="v2",
        mini_drag="dragdist",
        obstacle_colors=color_list,
        # obstacle_colors=[],
    )

    my_animation.run(save_animation=False)
    my_animation.logs(len(agent_list))


def run_ten_bed_animation_rviz(it_max: int = 800, go_to_center: bool = False):
    """Run environment using ros and rviz"""
    import rclpy
    from rclpy.node import Node
    from autonomous_furniture.rviz_animator import RvizSimulator

    agent_list = create_environment()

    ## Start ROS Node
    print("Starting publishing node")
    rclpy.init()
    visualizer = RvizSimulator(
        it_max=it_max,
        dt_simulation=0.01,
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


# import rclpy
# from rclpy.node import Node
# from autonomous_furniture.rviz_animator import RvizSimulator


if (__name__) == "__main__":
    # run_ten_bed_animation_matplotlib(it_max=200)
    run_ten_bed_animation_rviz(it_max=200)
