from typing import Optional

import numpy as np

import yaml
import matplotlib.pyplot as plt

from vartools.animator import Animator
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from autonomous_furniture.agent3D import Furniture3D
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from autonomous_furniture.agent_helper_functions import (
    update_multi_layer_simulation,
    plot_animation,
)


class DynamicalSystemAnimation3D(Animator):
    def __init__(
        self,
        parameter_file,
        it_max: int = None,
        iterator: Optional[int] = None,
        dt_simulation: float = None,
        dt_sleep: float = None,
        animation_name: str = "None",
        file_type: str = "None",
    ) -> None:
        with open(parameter_file, "r") as openfile:
            yaml_object = yaml.safe_load(openfile)

        if it_max == None:
            it_max = yaml_object["maximum simulation time"]

        if dt_simulation == None:
            dt_simulation = yaml_object["time step of simulation"]

        if dt_sleep == None:
            dt_sleep = yaml_object["time step of sleep"]

        if animation_name == "None":
            animation_name = yaml_object["animation name"]

        if file_type == "None":
            file_type = yaml_object["video file type"]

        super().__init__(
            it_max, iterator, dt_simulation, dt_sleep, animation_name, file_type
        )

        # For metrics
        self.metrics_json = {}
        # By default set to it_max, value is changed in metrics method
        self.it_final = it_max - 1

    def setup(
        self,
        layer_list: list[list[Furniture3D]],
        parameter_file: str,
        x_lim=None,
        y_lim=None,
        anim: bool = None,
        check_convergence: bool = None,
        obstacle_colors=None,
        figsize=None,
    ):
        with open(parameter_file, "r") as openfile:
            yaml_object = yaml.safe_load(openfile)

        if x_lim == None:
            self.x_lim = yaml_object["x limit"]
        else:
            self.x_lim = x_lim

        if y_lim == None:
            self.y_lim = yaml_object["y limit"]
        else:
            self.y_lim = y_lim

        if anim == None:
            anim = yaml_object["animation"]

        if obstacle_colors == None:
            self.obstacle_colors = yaml_object["obstacle colors"]
        else:
            self.obstacle_colors = obstacle_colors

        if check_convergence == None:
            self.check_convergence = yaml_object["check convergence"]
        else:
            self.check_convergence = check_convergence

        if figsize == None:
            figsize = yaml_object["figure size"]

        self.number_agent = len(layer_list[0])
        self.number_layer = len(layer_list)

        self.position_list = np.zeros((2, self.it_max))
        self.time_list = np.zeros((self.it_max))
        # self.position_list = [agent_list[ii]._reference_pose.position for ii in range(self.number_agent)]
        self.layer_list = layer_list

        self.agent_pos_saver = []
        for i in range(self.number_agent):
            self.agent_pos_saver.append([])
        for i in range(self.number_agent):
            saved = False
            for k in range(self.number_layer):
                if not self.layer_list[k] == None:
                    if (
                        not saved
                    ):  # make sure the positions are only saved once when an agent is present in multiple layers
                        self.agent_pos_saver[i].append(
                            self.layer_list[k][i]._reference_pose.position
                        )
                        saved = True

        if anim:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=120)

        self.converged: bool = False  # IF all the agent has converged

    def update_step(self, anim: bool = True):
        self.layer_list, self.agent_pos_saver = update_multi_layer_simulation(
            number_layer=self.number_layer,
            number_agent=self.number_agent,
            layer_list=self.layer_list,
            dt_simulation=self.dt_simulation,
            agent_pos_saver=self.agent_pos_saver,
        )

        if not anim:
            return

        plot_animation(
            self.ax,
            self.layer_list,
            self.number_layer,
            self.number_agent,
            self.obstacle_colors,
            self.agent_pos_saver,
            self.x_lim,
            self.y_lim,
        )

    def has_converged(self, it: int) -> bool:
        if not self.check_convergence:
            return False

        rtol_pos = 1e-3
        rtol_ang = 4e-1
        for ii in range(len(self.layer_list[0])):
            if not self.layer_list[0][ii].converged:
                if np.allclose(
                    self.layer_list[0][ii]._goal_pose.position,
                    self.layer_list[0][ii]._reference_pose.position,
                    rtol=rtol_pos,
                ) and np.allclose(
                    self.layer_list[0][ii]._goal_pose.orientation % np.pi,
                    self.layer_list[0][ii]._reference_pose.position % np.pi,
                    rtol=rtol_ang,
                ):
                    self.layer_list[0][ii].converged = True
                else:
                    return False

        self.converged = True  # All the agents has converged
        self.it_final = it + 1  # Because it starts at 0
        return True

    def logs(self, nb_furniture: int):
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
                self.metrics_json[f"agent_{ii}"].update(
                    {"prox": [1 - 1 / self.it_final * self.agent[ii]._proximity]}
                )
                self.metrics_json[f"agent_{ii}"].update(
                    {"list_prox": self.agent[ii]._list_prox}
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
                self.metrics_json[f"agent_{ii}"]["prox"].append(
                    1 - 1 / self.it_final * self.agent[ii]._proximity
                )

            if "total_dist" in self.metrics_json[f"agent_{ii}"]:
                self.metrics_json[f"agent_{ii}"]["total_dist"].append(
                    self.agent[ii].total_distance
                )
            else:
                self.metrics_json[f"agent_{ii}"]["total_dist"] = [
                    self.agent[ii].total_distance
                ]

        json_name = (
            "distance_"
            + f"nb{nb_furniture}_"
            + self.mini_drag
            + "_"
            + self.version
            + ".json"
        )
        with open(json_name, "w") as outfile:
            print(json.dump(self.metrics_json, outfile, indent=4))

    def run_no_clip(self, save_animation: bool = False) -> None:
        """Runs the without visualization
        --- this function has been recreated what I expected it to be..."""
        self.it_count = 0
        while self.it_max is None or self.it_count < self.it_max:
            self.update_step(self.it_count, anim=False)

            # Check convergence
            if self.has_converged(self.it_count):
                print(f"All trajectories converged at iteration={self.it_count}.")
                break

            self.it_count += 1
