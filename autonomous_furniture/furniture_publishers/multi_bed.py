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
from autonomous_furniture.furniture_creators import add_walls
from autonomous_furniture.evaluation.scenario_launcher import ScenarioLauncher


def create_environment(do_walls=True, do_person=True, n_agents: int = 10):
    import random

    x_lim: list[float] = [0, 12]
    y_lim: list[float] = [0, 9]
    # random.seed(7)
    # random.seed(3)
    random.seed(6)

    obstacle_environment = GlobalObstacleContainer()

    sample_bed = create_hospital_bed(
        start_pose=ObjectPose(position=np.zeros(2), orientation=0),
        margin_absolut=0.5,
        axes_length=np.array([2.2, 1.1]),
    )

    my_scenario = ScenarioLauncher(
        nb_furniture=n_agents,
        furniture_shape=sample_bed._shape,
        x_lim=x_lim,
        y_lim=y_lim,
    )
    # create goal poses
    goal_poses = []
    k = 0
    for i in range(n_agents):
        goal_orientation_i = np.pi / 2
        if (i % 2) == 0:  # divide agents in 2 rows
            goal_position_i = [
                x_lim[0] + (sample_bed._shape.axes_with_margin[1] / 2) * k,
                y_lim[0] + sample_bed._shape.axes_length[0] / 2,
            ]
        else:
            goal_position_i = [
                x_lim[0] + (sample_bed._shape.axes_with_margin[1] / 2) * k,
                y_lim[0]
                + sample_bed._shape.axes_with_margin[0]
                + sample_bed._shape.axes_length[0] / 2
                + 0.5,
            ]
            k += 1.5

        goal_i = ObjectPose(position=goal_position_i, orientation=goal_orientation_i)
        goal_poses.append(goal_i)

    my_scenario.creation(goal_poses)

    obstacle_environment.empty()

    agent_list: list[BaseAgent] = []
    for ii in range(n_agents):
        new_bed = create_hospital_bed(
            start_pose=my_scenario._init_setup[ii],
            goal_pose=my_scenario._goal_setup[ii],
            margin_absolut=0.5,
        )
        agent_list.append(new_bed)

    if do_person:
        start_position = [x_lim[1], y_lim[0]]
        goal_position = [x_lim[0], y_lim[1]]
        goal_pose = ObjectPose(position=goal_position)
        person = Person(
            center_position=start_position,
            priority_value=1e3,
            goal_pose=goal_pose,
            obstacle_environment=GlobalObstacleContainer.get(),
            margin=0.6,
            name="qolo_human",
        )
        agent_list.append(person)

    if do_walls:
        # add walls
        wall_width = 0.5
        area_enlargement = 1.5
        agent_list = add_walls(
            x_lim, y_lim, agent_list, wall_width, area_enlargement, wall_margin=0.7
        )

    return agent_list, x_lim, y_lim, wall_width, area_enlargement


def run_ten_bed_animation_matplotlib(it_max=800):
    # Get simple bed - to input it into the ScenarioLauncher
    import matplotlib.pyplot as plt
    from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation

    agent_list, x_lim, y_lim, wall_width, area_enlargement = create_environment(
        do_walls=True, do_person=False
    )

    n_agents = len(agent_list)
    if wall_width != None:
        n_agents -= 4  # take out the walls from the counting
    cm = plt.get_cmap("gist_rainbow")
    color_list = [cm(1.0 * ii / n_agents) for ii in range(n_agents)]
    if wall_width != None:
        for i in range(4):
            color_list.append("grey")

    my_animation = DynamicalSystemAnimation(
        it_max=it_max,
        dt_simulation=0.04,
        dt_sleep=0.04,
        animation_name=str(n_agents) + "_bed_animation",
        file_type=".gif",
    )

    total_enlargement = wall_width / 2 + area_enlargement
    x_lim_anim = [x_lim[0] - total_enlargement, x_lim[1] + total_enlargement]
    y_lim_anim = [y_lim[0] - total_enlargement, y_lim[1] + total_enlargement]

    my_animation.setup(
        obstacle_environment=GlobalObstacleContainer(),
        agent=agent_list,
        x_lim=x_lim_anim,
        y_lim=y_lim_anim,
        figsize=(10, 8),
        version="v2",
        mini_drag="dragdist",
        obstacle_colors=color_list,
        emergency_stop=True,
        safety_module=True,
        # obstacle_colors=[],
    )

    my_animation.run(save_animation=False)
    my_animation.logs(len(agent_list))


def run_ten_bed_animation_rviz(it_max: int = 1000, go_to_center: bool = False):
    """Run environment using ros and rviz"""
    import rclpy
    from rclpy.node import Node
    from autonomous_furniture.rviz_animator import RvizSimulator

    agent_list, x_lim, y_lim, wall_width, area_enlargement = create_environment(
        do_walls=True, do_person=True
    )

    ## Start ROS Node
    print("Starting publishing node")
    rclpy.init()
    visualizer = RvizSimulator(
        it_max=it_max,
        dt_simulation=0.01,
        dt_sleep=0.01,
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
    run_ten_bed_animation_matplotlib(it_max=1000)
    # run_ten_bed_animation_rviz(it_max=1000)
