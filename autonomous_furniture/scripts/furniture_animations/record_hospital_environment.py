import argparse
import time
from math import pi, cos, sin, sqrt

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from autonomous_furniture.furniture_class import (
    Furniture,
    FurnitureDynamics,
    FurnitureContainer,
    FurnitureAttractorDynamics,
)


class DynamicalSystemAnimation(Animator):
    dim = 2

    def setup(
        self,
        furniture_env,
        walls=False,
        x_lim=None,
        y_lim=None,
    ):
        num_obs = len(furniture_env)
        total_ctl_pts = 0
        for furniture in furniture_env:
            total_ctl_pts += furniture.num_control_points

        if y_lim is None:
            y_lim = [-3.0, 3.0]
        if x_lim is None:
            x_lim = [-3.0, 3.0]
        if walls is True:
            walls_center_position = np.array(
                [[0.0, y_lim[0]], [x_lim[0], 0.0], [0.0, y_lim[1]], [x_lim[1], 0.0]]
            )
            x_length = x_lim[1] - x_lim[0]
            y_length = y_lim[1] - y_lim[0]
            walls_size = [
                [x_length, 0.1],
                [0.1, y_length],
                [x_length, 0.1],
                [0.1, y_length],
            ]
            walls_orientation = [0, pi / 2, 0, pi / 2]
            for furniture in furniture_env:
                if furniture.furniture_type == "person":
                    wall_margin = furniture.get_margin()
            for ii in range(4):
                furniture_env.append(
                    Furniture(
                        "wall",
                        0,
                        walls_size[ii],
                        "Cuboid",
                        walls_center_position[ii],
                        walls_orientation[ii],
                    )
                )
        else:
            wall_margin = 0.0

        # max_axis = max(goals[0].axes_length)
        # min_axis = min(goals[0].axes_length)
        # offset = 0.5
        # wall_thickness = 0.3
        # parking_zone_cp = np.array([[x_lim[1] - (wall_margin + wall_thickness), y_lim[0] + (offset + max_axis / 2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness + min_axis + offset), y_lim[0] + (offset + max_axis / 2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness + 2 * (min_axis + offset)), y_lim[0] + (offset + max_axis / 2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness + 3 * (min_axis + offset)), y_lim[0] + (offset + max_axis / 2)]])
        # parking_zone = ObstacleContainer()
        # for pk in range(len(parking_zone_cp)):
        #     parking_zone.append(
        #         Cuboid(
        #             axes_length=goals[pk].axes_length,
        #             center_position=parking_zone_cp[pk],
        #             margin_absolut=0,
        #             orientation=pi / 2,
        #             tail_effect=False,
        #             repulsion_coeff=1,
        #             linear_velocity=np.array([0., 0.]),
        #         )
        #     )

        self.furniture_avoider = FurnitureDynamics(furniture_env)
        self.furniture_attractor_avoider = FurnitureAttractorDynamics(
            furniture_env, cutoff_distance=2
        )
        self.position_list = []
        self.velocity_list = []
        self.time_list = np.zeros((num_obs, self.it_max))

        for furniture in furniture_env:
            self.position_list.append(
                np.zeros((furniture.num_control_points, self.dim, self.it_max))
            )
            self.velocity_list.append(
                np.zeros((furniture.num_control_points, self.dim, self.it_max))
            )

        for ii, furniture in enumerate(furniture_env):
            if furniture.num_control_points > 0:
                self.position_list[ii][:, :, 0] = furniture.relative2global(
                    furniture.rel_ctl_pts_pos, furniture.furniture_container
                )

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.furniture_env = furniture_env

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        temp_pos = []
        for jj, _ in enumerate(self.furniture_env):
            temp_pos.append(self.position_list[jj][:, :, ii - 1])

        weights = self.furniture_avoider.get_influence_weight_at_points(temp_pos, 3)

        for jj, furniture in enumerate(self.furniture_env):
            if (
                furniture.furniture_type == "person"
                or furniture.furniture_type == "wall"
            ):
                continue

            global_attractor_position = furniture.relative2global(
                furniture.rel_ctl_pts_pos, furniture.goal_container
            )
            (
                goal_velocity,
                goal_rotation,
            ) = self.furniture_attractor_avoider.evaluate_furniture_attractor(
                global_attractor_position, jj
            )

            if furniture.attractor_state != "regroup":
                new_goal_position = (
                    goal_velocity * self.dt_simulation
                    + furniture.goal_container.center_position
                )
                new_goal_orientation = (
                    -(1 * goal_rotation * self.dt_simulation)
                    + furniture.goal_container.orientation
                )
            else:
                new_goal_position = furniture.parking_zone_position
                new_goal_orientation = furniture.parking_zone_orientation

            furniture.goal_container.center_position = new_goal_position
            furniture.goal_container.orientation = new_goal_orientation
            global_attractor_position = furniture.relative2global(
                furniture.rel_ctl_pts_pos, furniture.goal_container
            )
            self.furniture_avoider.set_attractor_position(global_attractor_position, jj)

        for jj, furniture in enumerate(self.furniture_env):
            if furniture.furniture_type == "wall":
                continue
            self.velocity_list[jj][
                :, :, ii
            ] = self.furniture_avoider.evaluate_furniture(
                self.position_list[jj][:, :, ii - 1], jj
            )

            furniture_lin_vel = np.zeros(2)

            for ctl_pt in range(furniture.num_control_points):
                furniture_lin_vel += (
                    self.velocity_list[jj][ctl_pt, :, ii] * weights[jj][ctl_pt]
                )

            ang_vel = np.zeros(furniture.num_control_points)

            for ctl_pt in range(furniture.num_control_points):
                ang_vel[ctl_pt] = weights[jj][ctl_pt] * np.cross(
                    furniture.furniture_container.center_position
                    - self.position_list[jj][ctl_pt, :, ii - 1],
                    self.velocity_list[jj][ctl_pt, :, ii] - furniture_lin_vel,
                )

            furniture_ang_vel = ang_vel.sum()

            if furniture.furniture_type != "person":
                furniture.furniture_container.linear_velocity = furniture_lin_vel
                furniture.furniture_container.angular_velocity = -2 * furniture_ang_vel
                furniture.furniture_container.do_velocity_step(self.dt_simulation)
            else:
                furniture.furniture_container.do_velocity_step(self.dt_simulation)

            if furniture.num_control_points > 0:
                self.position_list[jj][:, :, ii] = furniture.relative2global(
                    furniture.rel_ctl_pts_pos, furniture.furniture_container
                )

            # print(f"Max time: {max(time_list[obs, :])}, mean time: {sum(time_list[obs, :])/ii}, for obs: {obs}, with {len(obs_w_multi_agent[obs])} control points")

        self.ax.clear()

        # Drawing and adjusting of the axis
        obstacle_environment = self.furniture_env.generate_obstacle_environment()

        for jj, furniture in enumerate(self.furniture_env):
            plot_obstacles(
                ax=self.ax,
                obstacle_container=obstacle_environment,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                showLabel=False,
            )

            for ctl_pt in range(furniture.num_control_points):
                self.ax.plot(
                    self.position_list[jj][ctl_pt, 0, :ii],
                    self.position_list[jj][ctl_pt, 1, :ii],
                    ":",
                    color="#135e08",
                )
                self.ax.plot(
                    self.position_list[jj][ctl_pt, 0, ii],
                    self.position_list[jj][ctl_pt, 1, ii],
                    "o",
                    color="#135e08",
                    markersize=12,
                )
                self.ax.plot(
                    furniture.initial_dynamic[ctl_pt].attractor_position[0],
                    furniture.initial_dynamic[ctl_pt].attractor_position[1],
                    "k*",
                    markersize=8,
                )

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

    def has_converged(self, ii) -> bool:
        # return np.allclose(self.position_list[:, ii], self.position_list[:, ii - 1])
        return False


def run_multiple_furniture_avoiding_person():
    furniture_type = ["furniture", "furniture", "furniture", "furniture", "person"]
    num_ctl_furniture = [2, 2, 2, 2, 0]
    size_furniture = [[2.2, 1.1], [2.2, 1.1], [2.2, 1.1], [2.2, 1.1], [0.5, 0.5]]
    shape_furniture = ["Cuboid", "Cuboid", "Cuboid", "Cuboid", "Ellipse"]
    init_pos_furniture = [
        np.array([-1.5, 1.5]),
        np.array([-1.5, -1.5]),
        np.array([1.5, 1.5]),
        np.array([1.5, -1.5]),
        np.array([4.5, -1.2]),
    ]
    init_ori_furniture = [pi / 2, pi / 2, pi / 2, pi / 2, 0]
    init_vel_furniture = [
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([-0.3, 0.1]),
    ]
    goal_pos_furniture = [
        np.array([-1.5, 1.5]),
        np.array([-1.5, -1.5]),
        np.array([1.5, 1.5]),
        np.array([1.5, -1.5]),
        np.array([4.5, -1.2]),
    ]
    goal_ori_furniture = [pi / 2, pi / 2, pi / 2, pi / 2, 0]

    mobile_furniture = FurnitureContainer()

    for i in range(len(num_ctl_furniture)):
        mobile_furniture.append(
            Furniture(
                furniture_type[i],
                num_ctl_furniture[i],
                size_furniture[i],
                shape_furniture[i],
                init_pos_furniture[i],
                init_ori_furniture[i],
                init_vel_furniture[i],
                goal_pos_furniture[i],
                goal_ori_furniture[i],
            )
        )

    mobile_furniture.assign_margin()

    my_animation = DynamicalSystemAnimation(
        it_max=900,
        dt_simulation=0.05,
        dt_sleep=0.01,
        animation_name="full_env_rec",
    )

    my_animation.setup(
        furniture_env=mobile_furniture,
        walls=True,
        x_lim=[-6, 6],
        y_lim=[-5, 5],
    )

    my_animation.run(save_animation=args.rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec", action="store", default=False, help="Record flag")
    args = parser.parse_args()

    plt.close("all")
    plt.ion()

    run_multiple_furniture_avoiding_person()
