from typing import Optional

import numpy as np

import json
import matplotlib.pyplot as plt

from vartools.animator import Animator
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from autonomous_furniture.agent import BaseAgent


class DynamicalSystemAnimation(Animator):
    def __init__(
        self,
        it_max: int = 100,
        iterator: Optional[int] = None,
        dt_simulation: float = 0.1,
        dt_sleep: float = 0.1,
        animation_name: str = "",
        file_type=".mp4",
    ) -> None:
        super().__init__(
            it_max, iterator, dt_simulation, dt_sleep, animation_name, file_type
        )

        # For metrics
        self.metrics_json = {}
        # By default set to it_max, value is changed in metrics method
        self.it_final = it_max - 1

    def setup(
        self,
        obstacle_environment,
        agent: list[BaseAgent],
        x_lim=None,
        y_lim=None,
        anim: bool = True,
        mini_drag: str = "nodrag",
        version: str = "v1",
        safety_module: bool = True,
        emergency_stop: bool = True,
        check_convergence: bool = True,
        obstacle_colors=["orange", "blue", "red"],
        figsize=(3.0, 2.5),
    ):
        self.mini_drag = mini_drag
        self.version = version
        self.safety_module = safety_module
        self.emergency_stop = emergency_stop
        self.check_convergence = check_convergence

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

        self.agent_pos_saver = []
        for i in range(self.number_agent):
            self.agent_pos_saver.append([])
        for i in range(self.number_agent):
            self.agent_pos_saver[i].append(self.agent[i].position)

        self.obstacle_environment = obstacle_environment
        # for i in range(len(obstacle_environment)):
        #     self.obstacle_colors.append(np.array(np.random.choice(range(255),size=3))/254)
        self.obstacle_colors = obstacle_colors

        if anim:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=120)

        self.converged: bool = False  # IF all the agent has converged

    def update_step(self, ii, anim: bool = True):
        for jj in range(self.number_agent):
            self.agent[jj].update_velocity(
                mini_drag=self.mini_drag, version=self.version, emergency_stop=self.emergency_stop, safety_module=self.safety_module
            )
            self.agent[jj].compute_metrics(self.dt_simulation)
            self.agent[jj].do_velocity_step(self.dt_simulation)

            self.agent_pos_saver[jj].append(self.agent[jj].position)

        if not anim:
            return

        # print(f"Doing Step: {ii}")

        self.ax.clear()

        for jj in range(self.number_agent):
            global_control_points = self.agent[jj].get_global_control_points()
            self.ax.plot(global_control_points[0, :], global_control_points[1, :], "ko")

            goal_control_points = self.agent[
                jj
            ].get_goal_control_points()  ##plot agent center position

            if len(self.obstacle_colors) > jj:
                color = self.obstacle_colors[jj]
            else:
                color = "black"

            self.ax.plot(
                goal_control_points[0, :],
                goal_control_points[1, :],
                color=color,
                marker="o",
                linestyle="",  ##k=black, o=dot
            )

            # if self.agent[jj]._static == False:

            self.ax.plot(
                self.agent[jj]._goal_pose.position[0],
                self.agent[jj]._goal_pose.position[1],
                color=color,
                marker="*",
                markersize=10,
            )

            x_values = np.zeros(len(self.agent_pos_saver[jj]))
            y_values = x_values.copy()
            for i in range(len(self.agent_pos_saver[jj])):
                x_values[i] = self.agent_pos_saver[jj][i][0]
                y_values[i] = self.agent_pos_saver[jj][i][1]

            self.ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle="dashed",
            )

            # breakpoint()

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

        if len(self.obstacle_colors):
            for jj in range(self.number_agent):
                plot_obstacles(
                    ax=self.ax,
                    obstacle_container=[self.obstacle_environment[jj]],
                    x_lim=self.x_lim,
                    y_lim=self.y_lim,
                    showLabel=False,
                    obstacle_color=self.obstacle_colors[jj],
                    draw_reference=False,
                    set_axes=False,
                )
        else:
            plot_obstacles(
                ax=self.ax,
                obstacle_container=self.obstacle_environment,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                showLabel=False,
                obstacle_color=np.array([176, 124, 124]) / 255.0,
                draw_reference=False,
                set_axes=False,
            )

        self.ax.set_xlabel("x [m]", fontsize=9)
        self.ax.set_ylabel("y [m]", fontsize=9)

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        plt.tight_layout()

    def has_converged(self, it: int) -> bool:
        if not self.check_convergence:
            return False

        rtol_pos = 1e-3
        rtol_ang = 4e-1
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
