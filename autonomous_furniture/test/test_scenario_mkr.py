import numpy as np

import matplotlib.pyplot as plt
from scipy import rand
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose


from dynamic_obstacle_avoidance.visualization import plot_obstacles

from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation
from evaluation.scenario_launcher import ScenarioLauncher

import argparse

import json
import os
import random

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
parser.add_argument(
    "--name", action="store", default="recording", help="Name of the simulation"
)
args = parser.parse_args()


def multi_simulation(
    scenarios: list,
    nb_furniture: int,
    do_drag: bool,
    version: str = "v1",
    anim: bool = True,
):
    my_animation = DynamicalSystemAnimation(
        it_max=400,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_scenario = ScenarioLauncher(nb_furniture=nb_furniture)

    for ii in scenarios:
        random.seed(ii)
        my_scenario.creation()

        anim_name_pre = f"{args.name}_scen{ii}_nb{nb_furniture}_"

        anim_name = anim_name_pre + "drag" if do_drag else anim_name_pre + "no_drag_"
        anim_name += version
        my_animation.animation_name = anim_name

        my_scenario.setup()

        my_animation.setup(
            my_scenario.obstacle_environment,
            agent=my_scenario.agents,
            x_lim=[-3, 8],
            y_lim=[-2, 7],
            anim=anim,
        )

        print(
            f"Number of fur  : {nb_furniture} | Alg with drag : {do_drag} | version : {version} | Number of fold : {ii}"
        )
        if anim:
            # save_animation=args.rec,
            my_animation.run(
                save_animation=args.rec, mini_drag=do_drag, version=version
            )
        else:
            # save_animation=args.rec,
            my_animation.run_no_clip(mini_drag=do_drag, version=version)

        my_animation.logs(nb_furniture, do_drag, version=version)
        # Reset of the collisions counter (TODO: To be changed it's ugly)
        BaseAgent.number_collisions = 0
        BaseAgent.number_serious_collisions = 0


def single_simulation(
    scen: int, nb_furniture: int, do_drag: bool, version: str = "v1", anim: bool = True
):
    my_animation = DynamicalSystemAnimation(
        it_max=290,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )

    random.seed(scen)

    my_scenario = ScenarioLauncher(nb_furniture=nb_furniture)
    my_scenario.creation()
    anim_name_pre = f"{args.name}_scen{scen}_"

    anim_name = anim_name_pre + "drag" if do_drag else anim_name_pre + "no_drag"
    anim_name += version
    my_animation.animation_name = anim_name

    my_scenario.setup()

    my_animation.setup(
        my_scenario.obstacle_environment,
        agent=my_scenario.agents,
        x_lim=[-3, 8],
        y_lim=[-2, 7],
        anim=anim,
    )
    print(
        f"Number of fur  : {nb_furniture} | Alg with drag : {do_drag} | version : {version} | Number of fold : {scen}"
    )

    if anim:
        my_animation.run(save_animation=args.rec, mini_drag=do_drag, version=version)
    else:
        my_animation.run_no_clip(mini_drag=do_drag, version=version)

    my_animation.logs(nb_furniture, do_drag, version=version)


def main():
    # List of environment shared by all the furniture/agent
    scenarios = range(100)

    for nb_furniture in [2]:
        for version in ["v2", "v1"]:
            for do_drag in [True]:
                multi_simulation(
                    scenarios, nb_furniture, do_drag, version=version, anim=False
                )


def run_single():
    scen = 50
    nb_furniture = 4
    version = "v2"
    for do_drag in [False]:
        single_simulation(scen, nb_furniture, do_drag, version=version, anim=True)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    # main()
    run_single()
