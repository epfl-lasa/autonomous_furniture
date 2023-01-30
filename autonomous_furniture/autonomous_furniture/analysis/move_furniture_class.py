import numpy as np
from math import pi
import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from autonomous_furniture.furniture_class import (
    Furniture,
    FurnitureDynamics,
    FurnitureContainer,
    FurnitureAttractorDynamics,
)


class DynamicFurniture:
    dim = 2

    def __init__(self):
        self.animation_paused = True

    def on_click(self, event):
        self.animation_paused = not self.animation_paused

    def run(
        self,
        furniture_env,
        walls=False,
        x_lim=None,
        y_lim=None,
        it_max=1000,
        dt_step=0.03,
        dt_sleep=0.1,
    ):

        num_obs = len(furniture_env)
        total_ctl_pts = 0
        for furniture in furniture_env:
            total_ctl_pts += furniture.num_control_points

        print(total_ctl_pts)

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
            walls_orientaion = [0, pi / 2, 0, pi / 2]
            wall_margin = furniture_env[-1].get_margin()
            for ii in range(4):
                furniture_env.append(
                    Furniture(
                        "wall",
                        0,
                        walls_size[ii],
                        "Cuboid",
                        walls_center_position[ii],
                        walls_orientaion[ii],
                    )
                )
        else:
            wall_margin = 0.0

        parking_zone_cp = np.array([0, 0])
        parking_zone = ObstacleContainer()
        for pk in range(len(parking_zone_cp)):
            parking_zone.append(
                Cuboid(
                    axes_length=furniture_env[0].goal_container.axes_length,
                    center_position=parking_zone_cp,
                    margin_absolut=0,
                    orientation=pi / 2,
                    tail_effect=False,
                    repulsion_coeff=1,
                )
            )

        furniture_avoider = FurnitureDynamics(furniture_env)
        furniture_attractor_avoider = FurnitureAttractorDynamics(
            furniture_env, cutoff_distance=2.2
        )
        position_list = []
        velocity_list = []

        for furniture in furniture_env:
            position_list.append(
                np.zeros((furniture.num_control_points, self.dim, it_max))
            )
            velocity_list.append(
                np.zeros((furniture.num_control_points, self.dim, it_max))
            )

        for ii, furniture in enumerate(furniture_env):
            if furniture.num_control_points > 0:
                position_list[ii][:, :, 0] = furniture.relative2global(
                    furniture.rel_ctl_pts_pos, furniture.furniture_container
                )

        fig, ax = plt.subplots(figsize=(10, 8))  # figsize=(10, 8)
        ax.set_aspect(1.0)
        cid = fig.canvas.mpl_connect("button_press_event", self.on_click)

        print(furniture_env)

        ii = 0
        while ii < it_max:
            if self.animation_paused:
                plt.pause(dt_sleep)
                if not plt.fignum_exists(fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                continue

            ii += 1
            if ii > it_max:
                break

            temp_pos = []
            for jj, _ in enumerate(furniture_env):
                temp_pos.append(position_list[jj][:, :, ii - 1])

            weights = furniture_avoider.get_influence_weight_at_points(temp_pos, 3)
            # print(f"weights: {weights}")

            for jj, furniture in enumerate(furniture_env):
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
                ) = furniture_attractor_avoider.evaluate_furniture_attractor(
                    global_attractor_position, jj
                )

                if jj == 0:
                    # print(f"state of furn: \n {furniture.attractor_state} \n")
                    pass

                if furniture.attractor_state != "regroup":
                    new_goal_position = (
                        goal_velocity * dt_step
                        + furniture.goal_container.center_position
                    )
                    new_goal_orientation = (
                        -(1 * goal_rotation * dt_step)
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
                furniture_avoider.set_attractor_position(global_attractor_position, jj)

            for jj, furniture in enumerate(furniture_env):
                if furniture.furniture_type == "wall":
                    continue
                velocity_list[jj][:, :, ii] = furniture_avoider.evaluate_furniture(
                    position_list[jj][:, :, ii - 1], jj
                )

                furniture_lin_vel = np.zeros(2)

                for ctl_pt in range(furniture.num_control_points):
                    furniture_lin_vel += (
                        velocity_list[jj][ctl_pt, :, ii] * weights[jj][ctl_pt]
                    )

                ang_vel = np.zeros(furniture.num_control_points)

                for ctl_pt in range(furniture.num_control_points):
                    ang_vel[ctl_pt] = weights[jj][ctl_pt] * np.cross(
                        furniture.furniture_container.center_position
                        - position_list[jj][ctl_pt, :, ii - 1],
                        velocity_list[jj][ctl_pt, :, ii] - furniture_lin_vel,
                    )

                furniture_ang_vel = ang_vel.sum()

                if furniture.furniture_type != "person":
                    furniture.furniture_container.linear_velocity = furniture_lin_vel
                    furniture.furniture_container.angular_velocity = (
                        -2 * furniture_ang_vel
                    )
                    furniture.furniture_container.do_velocity_step(dt_step)
                else:
                    furniture.furniture_container.do_velocity_step(dt_step)

                if furniture.num_control_points > 0:
                    position_list[jj][:, :, ii] = furniture.relative2global(
                        furniture.rel_ctl_pts_pos, furniture.furniture_container
                    )

            ax.clear()

            obstacle_environment = furniture_env.generate_obstacle_environment()

            for jj, furniture in enumerate(furniture_env):
                plot_obstacles(
                    ax,
                    obstacle_environment,
                    x_lim,
                    y_lim,
                    showLabel=False,
                )

                for ctl_pt in range(furniture.num_control_points):
                    ax.plot(
                        position_list[jj][ctl_pt, 0, :ii],
                        position_list[jj][ctl_pt, 1, :ii],
                        ":",
                        color="#135e08",
                    )
                    ax.plot(
                        position_list[jj][ctl_pt, 0, ii],
                        position_list[jj][ctl_pt, 1, ii],
                        "o",
                        color="#135e08",
                        markersize=12,
                    )
                    ax.plot(
                        furniture.initial_dynamic[ctl_pt].attractor_position[0],
                        furniture.initial_dynamic[ctl_pt].attractor_position[1],
                        "k*",
                        markersize=8,
                    )

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.grid()
            ax.set_aspect("equal", adjustable="box")

            plt.pause(dt_sleep)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break


def single_smart_furniture():
    furniture_type = ["furniture", "furniture", "furniture", "person"]
    num_ctl_furniture = [(3, 2), 2, 2, 0]
    size_furniture = [[2, 1], [2, 1], [2, 1], [0.4, 0.4]]
    shape_furniture = ["Cuboid", "Cuboid", "Cuboid", "Ellipse"]
    init_pos_furniture = [
        np.array([-2, 2]),
        np.array([-2, -2]),
        np.array([2, -2]),
        np.array([2, 2]),
    ]
    init_ori_furniture = [pi / 2, pi / 2, pi / 2, 0.0]
    init_vel_furniture = [
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([-0.5, -0.5]),
    ]
    goal_pos_furniture = [
        np.array([1, 1]),
        np.array([2, -0.9]),
        np.array([-1, -2]),
        np.array([1, -1]),
    ]
    goal_ori_furniture = [0.0, 0.0, 0.0, 0.0]

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

    print(mobile_furniture)

    margins = mobile_furniture.assign_margin()

    print(margins)

    DynamicFurniture().run(
        furniture_env=mobile_furniture,
        walls=True,
        x_lim=[-5, 5],
        y_lim=[-4, 4],
    )


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    single_smart_furniture()
