from math import sin, cos, pi
import time
import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from vartools.dynamical_systems import LinearSystem

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformBroadcaster, TransformStamped

from autonomous_furniture.furniture_class import (
    Furniture,
    FurnitureDynamics,
    FurnitureContainer,
    FurnitureAttractorDynamics,
)


def euler_to_quaternion(roll, pitch, yaw):
    qx = sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2) - cos(roll / 2) * sin(
        pitch / 2
    ) * sin(yaw / 2)
    qy = cos(roll / 2) * sin(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * cos(
        pitch / 2
    ) * sin(yaw / 2)
    qz = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) - sin(roll / 2) * sin(
        pitch / 2
    ) * cos(yaw / 2)
    qw = cos(roll / 2) * cos(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * sin(
        pitch / 2
    ) * sin(yaw / 2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def rotate_vector(vector, angle):
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    new_dir = np.dot(rot, vector)
    return new_dir


class DynamicalSystemRviz(Node):
    dim = 2

    def __init__(self):
        self.animation_paused = True
        rclpy.init()
        super().__init__("DS_state_publisher")
        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()
        self.odom_trans = TransformStamped()
        self.odom_trans.header.frame_id = "odom"

    def __del__(self):
        rclpy.shutdown()

    def on_click(self, event):
        self.animation_paused = not self.animation_paused

    def update_state_publisher(self, prefix, position, rotation):
        self.odom_trans.child_frame_id = prefix + "base_link"
        now = self.get_clock().now()
        self.odom_trans.header.stamp = now.to_msg()
        self.odom_trans.transform.translation.x = position[0]
        self.odom_trans.transform.translation.y = position[1]
        self.odom_trans.transform.translation.z = 0.0
        self.odom_trans.transform.rotation = euler_to_quaternion(
            pi / 2, 0, rotation
        )  # rpy
        if "qolo" in prefix:
            self.odom_trans.transform.translation.z = 0.2
            self.odom_trans.transform.rotation = euler_to_quaternion(
                0, 0, rotation
            )  # rpy

        # send the joint state and transform
        self.broadcaster.sendTransform(self.odom_trans)

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
        loop_rate = self.create_rate(30)

        num_obs = len(furniture_env)
        total_ctl_pts = 0
        for furniture in furniture_env:
            total_ctl_pts += furniture.num_control_points

        if y_lim is None:
            y_lim = [-0.5, 2.5]
        if x_lim is None:
            x_lim = [-1.5, 2]
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

        # x_offset = 1.5
        # y_offset = 1.
        # parking_zone_cp = np.array([[1. + x_offset, -.42 + y_offset],
        #                             [-1. + x_offset, -.42 + y_offset],
        #                             [1. + x_offset, .42 + y_offset],
        #                             [-1 + x_offset, .42 + y_offset],
        #                             [0. + x_offset, 0. + y_offset]])
        # parking_zone_or = [pi,
        #                    0.,
        #                    pi,
        #                    0.,
        #                    pi / 2]
        # parking_zone = ObstacleContainer()
        # for pk in range(len(parking_zone_cp)):
        #     parking_zone.append(
        #         Cuboid(
        #             axes_length=goals[pk].axes_length,
        #             center_position=parking_zone_cp[pk],
        #             margin_absolut=0,
        #             orientation=parking_zone_or[pk],
        #             tail_effect=False,
        #             repulsion_coeff=1,
        #             linear_velocity=np.array([0., 0.]),
        #         )
        #     )

        furniture_avoider = FurnitureDynamics(furniture_env)
        furniture_attractor_avoider = FurnitureAttractorDynamics(
            furniture_env, cutoff_distance=2
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

        obs_name = ["h_bed_1_", "h_bed_2_", "h_bed_3_", "h_bed_4_", "qolo_human_"]

        fig, ax = plt.subplots()  # figsize=(10, 8)
        cid = fig.canvas.mpl_connect("button_press_event", self.on_click)
        ax.set_aspect(1.0)

        ii = 0
        while ii < it_max:
            rclpy.spin_once(self)

            if self.animation_paused:
                plt.pause(dt_sleep)
                if not plt.fignum_exists(fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                continue

            ii += 1
            if ii > it_max:
                break

            # Here come the main calculation part
            temp_pos = []
            for jj, _ in enumerate(furniture_env):
                temp_pos.append(position_list[jj][:, :, ii - 1])

            weights = furniture_avoider.get_influence_weight_at_points(temp_pos, 3)

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

            for index, furniture in enumerate(furniture_env):
                if furniture.furniture_type == "person":
                    u_obs_vel = (
                        furniture.furniture_container.linear_velocity
                        / np.linalg.norm(furniture.furniture_container.linear_velocity)
                    )
                    x_vec = np.array([1, 0])
                    dot_prod = np.dot(x_vec, u_obs_vel)
                    qolo_dir = np.arccos(dot_prod)
                    self.update_state_publisher(
                        obs_name[index],
                        furniture.furniture_container.center_position,
                        qolo_dir,
                    )
                elif furniture.furniture_type == "furniture":
                    self.update_state_publisher(
                        obs_name[index],
                        furniture.furniture_container.center_position,
                        furniture.furniture_container.orientation,
                    )

            loop_rate.sleep()

            # Clear right before drawing again
            ax.clear()

            obstacle_environment = furniture_env.generate_obstacle_environment()

            # Drawing and adjusting of the axis
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


def main():
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

    DynamicalSystemRviz().run(
        furniture_env=mobile_furniture,
        walls=True,
        x_lim=[-6, 6],
        y_lim=[-5, 5],
        it_max=1500,
        dt_step=0.03,
        dt_sleep=0.01,
    )


if __name__ == "__main__":
    plt.close("all")

    try:
        main()
    except RuntimeError:
        pass
        try:
            rclpy.shutdown()
        except rclpy.RCLError:
            pass
