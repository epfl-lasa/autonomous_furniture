import numpy as np

import matplotlib.pyplot as plt
from scipy import rand
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose


from dynamic_obstacle_avoidance.visualization import plot_obstacles

from autonomous_furniture.agent import BaseAgent
from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation
from autonomous_furniture.evaluation.scenario_launcher import ScenarioLauncher

import argparse

import json
import os
import random

from multiprocessing import Process

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
parser.add_argument(
    "--name", action="store", default="recording", help="Name of the simulation"
)
args = parser.parse_args()


def multi_simulation(
    scenarios: list,
    nb_furniture: int,
    do_drag: str,
    version: str,
    anim: bool,
    it_max: int,
):
    my_animation = DynamicalSystemAnimation(
        it_max=it_max,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_scenario = ScenarioLauncher(nb_furniture=nb_furniture)

    for ii in scenarios:
        random.seed(ii)
        my_animation.it_final = (
            my_animation.it_max
        )  # We need to reset to this value to compute correctly the proximity
        my_scenario.creation()

        anim_name_pre = f"{args.name}_scen{ii}_nb{nb_furniture}_"

        anim_name = anim_name_pre + do_drag
        anim_name += version
        my_animation.animation_name = anim_name

        my_scenario.setup()

        my_animation.setup(
            my_scenario.obstacle_environment,
            agent=my_scenario.agents,
            x_lim=[-3, 8],
            y_lim=[-2, 7],
            anim=anim,
            mini_drag=do_drag,
            version=version,
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
            my_animation.run_no_clip()

        my_animation.logs(nb_furniture)
        # Reset of the collisions counter (TODO: To be changed it's ugly)
        BaseAgent.number_collisions = 0
        BaseAgent.number_serious_collisions = 0


def single_simulation(
    scen: int, nb_furniture: int, do_drag: str, version: str = "v1", anim: bool = True
):
    my_animation = DynamicalSystemAnimation(
        it_max=500,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )

    random.seed(scen)
    my_animation.it_final = (
        my_animation.it_max
    )  # We need to reset to this value to compute correctly the proximity

    my_scenario = ScenarioLauncher(nb_furniture=nb_furniture)
    my_scenario.creation()
    anim_name_pre = f"{args.name}_scen{scen}_"

    anim_name = anim_name_pre + do_drag
    anim_name += version
    my_animation.animation_name = anim_name

    my_scenario.setup()

    my_animation.setup(
        my_scenario.obstacle_environment,
        agent=my_scenario.agents,
        x_lim=[-3, 8],
        y_lim=[-2, 7],
        anim=anim,
        mini_drag=do_drag,
        version=version,
    )
    print(
        f"Number of fur  : {nb_furniture} | Alg with drag : {do_drag} | version : {version} | Number of fold : {scen}"
    )

    if anim:
        my_animation.run(save_animation=args.rec)
    else:
        my_animation.run_no_clip()

    my_animation.logs(nb_furniture)


def main():
    # List of environment shared by all the furniture/agent
    scenarios = range(150, 250)
    nb_array = [3]
    version_array = ["v2"]
    drag_array = ["nodrag"]

    n_process = len(nb_array) * len(version_array) * len(drag_array)
    process_array = []

    for nb_furniture in nb_array:
        for version in version_array:
            for do_drag in drag_array:
                # p = Process(multi_simulation, args=([scenarios, nb_furniture, do_drag, 1000, version, False]))
                # process_array.append(p)
                # p.start()
                # p.join()
                multi_simulation(
                    scenarios,
                    nb_furniture,
                    do_drag,
                    it_max=1000,
                    version=version,
                    anim=False,
                )
    # for i in range(n_process):
    #     p = process_array[i]
    #     p.start()
    # for i in range(n_process):
    #     p = process_array[i]
    #     p.join()


def run_single():
    scen = 240
    nb_furniture = 10
    version = "v2"
    for do_drag in ["dragdist"]:
        single_simulation(scen, nb_furniture, do_drag, version=version, anim=True)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    # main()
    run_single()
