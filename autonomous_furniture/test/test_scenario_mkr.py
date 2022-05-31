from math import pi, cos, sin, sqrt

import numpy as np

import matplotlib.pyplot as plt
from scipy import rand
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from autonomous_furniture.attractor_dynamics import AttractorDynamics, main

from autonomous_furniture.agent import BaseAgent, Furniture, Person
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


class DynamicalSystemAnimation(Animator):
    def __init__(
        self,
        it_max: int = 100,
        iterator=None,
        dt_simulation: float = 0.1,
        dt_sleep: float = 0.1,
        animation_name=None,
        file_type=".mp4",
    ) -> None:
        super().__init__(
            it_max, iterator, dt_simulation, dt_sleep, animation_name, file_type
        )

        # For metrics
        self.metrics_json = {}

    def setup(
        self,
        obstacle_environment,
        agent: BaseAgent,
        x_lim=None,
        y_lim=None,
        anim: bool = True,
    ):

        dim = 2
        self.number_agent = len(agent)

        if y_lim is None:
            y_lim = [0.0, 10]
        if x_lim is None:
            x_lim = [0, 10]

        self.position_list = np.zeros((dim, self.it_max))
        self.time_list = np.zeros((self.it_max))
        self.position_list = [agent[ii].position for ii in range(self.number_agent)]
        self.agent = agent
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment

        if anim:
            self.fig, self.ax = plt.subplots()

        self.converged: bool = False  # IF all the agent has converged

    def update_step(
        self, ii, mini_drag: bool = True, anim: bool = True, version: str = "v1"
    ):
        if not ii % 10:
            print(f"it={ii}")

        if anim:
            self.ax.clear()
            # Drawing and adjusting of the axis
            # for agent in range(self.num_agent):
            #     self.ax.plot(
            #         self.position_list[agent, 0, :ii + 1],
            #         self.position_list[agent, 1, :ii + 1],
            #         ":",
            #         color="#135e08"
            #     )
            #     self.ax.plot(
            #         self.position_list[agent, 0, ii + 1],
            #         self.position_list[agent, 1, ii + 1],
            #         "o",
            #         color="#135e08",
            #         markersize=12,
            #     )

            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim)

            plot_obstacles(
                self.ax,
                self.obstacle_environment,
                self.x_lim,
                self.y_lim,
                showLabel=False,
            )
        for jj in range(self.number_agent):
            self.agent[jj].update_velocity(mini_drag=mini_drag, version=version)
            self.agent[jj].compute_metrics(self.dt_simulation)
            self.agent[jj].do_velocity_step(self.dt_simulation)

            if anim:
                global_crontrol_points = self.agent[jj].get_global_control_points()
                self.ax.plot(
                    global_crontrol_points[0, :], global_crontrol_points[1, :], "ko"
                )

                goal_crontrol_points = self.agent[jj].get_goal_control_points()
                self.ax.plot(
                    goal_crontrol_points[0, :], goal_crontrol_points[1, :], "ko"
                )

                # for agent in range(self.num_agent):
                #     plt.arrow(self.position_list[agent, 0, ii + 1],
                #               self.position_list[agent, 1, ii + 1],
                #               self.velocity[agent, 0],
                #               self.velocity[agent, 1],
                #               head_width=0.05,
                #               head_length=0.1,
                #               fc='k',
                #               ec='k')

                #     self.ax.plot(
                #         self.initial_dynamics[agent].attractor_position[0],
                #         self.initial_dynamics[agent].attractor_position[1],
                #         "k*",
                #         markersize=8,
                #     )
                # self.ax.grid()
                self.ax.set_aspect("equal", adjustable="box")

    def has_converged(self) -> bool:
        rtol_pos = 1e-3
        rtol_ang = 1e-1
        for ii in range(len(self.agent)):
            if not self.agent[ii].converged:
                if np.allclose(
                    self.agent[ii]._goal_pose.position,
                    self.agent[ii].position,
                    rtol=rtol_pos,
                ) and np.allclose(
                    self.agent[ii]._goal_pose.orientation % np.pi,
                    self.agent[ii].orientation % np.pi,
                    rtol=rtol_ang,
                ):
                    self.agent[ii].converged = True
                else:
                    return False

        self.converged = True  # All the agents has converged
        return True

    def logs(self, nb_furniture: int, do_drag: bool, version: str = "v1"):
        if (
            self.metrics_json == {}
        ):  # If this is the first time we enter the parameters of the simulation
            self.metrics_json["max_step"] = self.it_max
            self.metrics_json["dt"] = self.dt_simulation
            self.metrics_json["converged"] = [self.converged]
            self.metrics_json["collisions"] = [BaseAgent.number_collisions]
            self.metrics_json["collisions_ser"] = [BaseAgent.number_serious_collisions]
        else:
            self.metrics_json["converged"].append(self.converged)
            self.metrics_json["collisions"].append(BaseAgent.number_collisions)
            self.metrics_json["collisions_ser"].append(
                BaseAgent.number_serious_collisions
            )

        for ii in range(len(self.agent)):
            if not f"agent_{ii}" in self.metrics_json:
                self.metrics_json.update({f"agent_{ii}": {}})
                self.metrics_json[f"agent_{ii}"].update({"id": ii})
                self.metrics_json[f"agent_{ii}"].update(
                    {"time_conv": [self.agent[ii].time_conv]}
                )
                self.metrics_json[f"agent_{ii}"].update(
                    {"time_direct": [self.agent[ii].time_conv_direct]}
                )
                self.metrics_json[f"agent_{ii}"].update(
                    {"direct_dist": [self.agent[ii].direct_distance]}
                )
            else:
                self.metrics_json[f"agent_{ii}"]["direct_dist"].append(
                    self.agent[ii].direct_distance
                )
                self.metrics_json[f"agent_{ii}"]["time_conv"].append(
                    self.agent[ii].time_conv
                )
                self.metrics_json[f"agent_{ii}"]["time_direct"].append(
                    self.agent[ii].time_conv_direct
                )

            if "total_dist" in self.metrics_json[f"agent_{ii}"]:
                self.metrics_json[f"agent_{ii}"]["total_dist"].append(
                    self.agent[ii].total_distance
                )
            else:
                self.metrics_json[f"agent_{ii}"]["total_dist"] = [
                    self.agent[ii].total_distance
                ]

        do_drag_str = "drag" if do_drag else "nodrag"
        json_name = (
            "distance_" + f"nb{nb_furniture}_" + f"{do_drag_str}_" + version + ".json"
        )
        with open(json_name, "w") as outfile:
            print(json.dump(self.metrics_json, outfile, indent=4))


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
    scen = 78
    nb_furniture = 2
    version = "v2"
    for do_drag in [True]:
        single_simulation(scen, nb_furniture, do_drag, version=version, anim=True)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    # main()
    run_single()
