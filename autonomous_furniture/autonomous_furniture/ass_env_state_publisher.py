import math
from math import sin, cos, pi
import time
import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from vartools.dynamical_systems import LinearSystem

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformBroadcaster, TransformStamped

from calc_time import calculate_relative_position, relative2global, global2relative
from autonomous_furniture.attractor_dynamics import AttractorDynamics


def euler_to_quaternion(roll, pitch, yaw):
    qx = sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2) - cos(roll / 2) * sin(pitch / 2) * sin(yaw / 2)
    qy = cos(roll / 2) * sin(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * cos(pitch / 2) * sin(yaw / 2)
    qz = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) - sin(roll / 2) * sin(pitch / 2) * cos(yaw / 2)
    qw = cos(roll / 2) * cos(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * sin(pitch / 2) * sin(yaw / 2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def rotate_vector(vector, angle):
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    new_dir = np.dot(rot, vector)
    return new_dir


class DynamicalSystemRviz(Node):
    def __init__(self):
        self.animation_paused = False
        rclpy.init()
        super().__init__('DS_state_publisher')
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
        self.odom_trans.transform.translation.z = 0.
        self.odom_trans.transform.rotation = \
            euler_to_quaternion(0, 0, rotation)  # rpy
        if "table" in prefix:
            self.odom_trans.transform.translation.x = position[0] + 0.2
        if "qolo" in prefix:
            self.odom_trans.transform.translation.z = 0.2
            self.odom_trans.transform.rotation = \
                euler_to_quaternion(0, 0, rotation)  # rpy

        # send the joint state and transform
        self.broadcaster.sendTransform(self.odom_trans)

    def run(
            self,
            initial_dynamics,
            obstacle_environment,
            obs_w_multi_agent,
            start_position=None,
            relative_attractor_position=None,
            goals=None,
            walls=False,
            x_lim=None,
            y_lim=None,
            it_max=1000,
            dt_step=0.03,
            dt_sleep=0.1
    ):
        loop_rate = self.create_rate(30)

        degree = pi / 180.0
        angle = 0.

        num_obs = len(obstacle_environment)
        if start_position.ndim > 1:
            num_agent = len(start_position)
        else:
            num_agent = 1
        dim = 2

        if y_lim is None:
            y_lim = [-0.5, 2.5]
        if x_lim is None:
            x_lim = [-1.5, 2]
        if start_position is None:
            start_position = np.zeros((num_obs, dim))
        if num_agent > 1:
            velocity = np.zeros((num_agent, dim))
        else:
            velocity = np.zeros((2, dim))
        if relative_attractor_position is None:
            relative_attractor_position = np.array([0., 0.])
        if goals is None:
            goals = Cuboid(
                axes_length=[0.6, 0.6],
                center_position=np.array([0., 0.]),
                margin_absolut=0,
                orientation=0,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0., 0.]),
            )
        if walls is True:
            walls_center_position = np.array([[0., y_lim[0]], [x_lim[0], 0.], [0., y_lim[1]], [x_lim[1], 0.]])
            x_length = x_lim[1] - x_lim[0]
            y_length = y_lim[1] - y_lim[0]
            wall_margin = obstacle_environment[-1].margin_absolut
            walls_cont = [
                Cuboid(
                    axes_length=[x_length, 0.1],  # [x_length, y_length],
                    center_position=walls_center_position[0],  # np.array([0., 0.]),
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                    is_boundary=False,
                ),
                Cuboid(
                    axes_length=[0.1, y_length],
                    center_position=walls_center_position[1],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                ),
                Cuboid(
                    axes_length=[x_length, 0.1],
                    center_position=walls_center_position[2],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                ),
                Cuboid(
                    axes_length=[0.1, y_length],
                    center_position=walls_center_position[3],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                )
            ]
            obstacle_environment = obstacle_environment[:-1] + walls_cont + obstacle_environment[-1:]
        else:
            wall_margin = 0.

        max_axis = max(goals[0].axes_length)
        min_axis = min(goals[0].axes_length)
        offset = 0.5
        wall_thickness = 0.3
        # parking_zone_cp = np.array([[x_lim[1] - (wall_margin + wall_thickness), y_lim[0] + (offset + max_axis / 2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness + min_axis + offset), y_lim[0] + (offset + max_axis / 2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness + 2 * (min_axis + offset)), y_lim[0] + (offset + max_axis / 2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness + 3 * (min_axis + offset)), y_lim[0] + (offset + max_axis / 2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness + 4 * (min_axis + offset)), y_lim[0] + (offset + max_axis / 2)]])
        x_offset = 1.5
        y_offset = -1.
        parking_zone_cp = np.array([[1. + x_offset, -.5 + y_offset],
                                    [-1. + x_offset, -.5 + y_offset],
                                    [1. + x_offset, .5 + y_offset],
                                    [-1 + x_offset, .5 + y_offset],
                                    [0. + x_offset, 0. + y_offset]])
        parking_zone_or = [pi,
                           0.,
                           pi,
                           0.,
                           pi / 2]
        parking_zone = ObstacleContainer()
        for pk in range(len(parking_zone_cp)):
            parking_zone.append(
                Cuboid(
                    axes_length=goals[pk].axes_length,
                    center_position=parking_zone_cp[pk],
                    margin_absolut=0,
                    orientation=parking_zone_or[pk],
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0., 0.]),
                )
            )

        attractor_dynamic = AttractorDynamics(obstacle_environment, cutoff_dist=2, parking_zone=parking_zone)
        dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=initial_dynamics, environment=obstacle_environment, obs_multi_agent=obs_w_multi_agent)
        position_list = np.zeros((num_agent, dim, it_max))
        time_list = np.zeros((num_obs, it_max))
        relative_agent_pos = np.zeros((num_agent, dim))

        # for obs, (name, clt_pts) in enumerate(obs_w_multi_agent.items()):
        #     relative_agent_pos = - (obstacle_environment[name].center_position - start_position)

        for obs in range(num_obs):
            relative_agent_pos[obs_w_multi_agent[obs], :] = global2relative(start_position[obs_w_multi_agent[obs]],
                                                                            obstacle_environment[obs])
            # for agent in obs_w_multi_agent[obs]:
            #     if start_position.ndim > 1:
            #         relative_agent_pos[agent, :] = - (obstacle_environment[obs].center_position - start_position[agent, :])
            #     else:
            #         relative_agent_pos = - (obstacle_environment[obs].center_position - start_position)

        obs_name = ["h_bed_1_", "h_bed_2_", "h_bed_3_", "h_bed_4_", "qolo_"]
        obs_name = ["chair_1_", "chair_2_", "chair_3_", "chair_4_", "table_", "qolo_"]
        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots()  # figsize=(10, 8)
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)
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
            weights = dynamic_avoider.get_influence_weight_at_ctl_points(position_list[:, :, ii-1], 3)
            # print(f"weights: {weights}")

            # if ii % 2 == 0:
            for jj, goal in enumerate(goals):
                # print(f"     goal: {goal}")              # make a class instead of the dictionary
                num_attractor = len(obs_w_multi_agent[jj])
                global_attractor_pos = relative2global(relative_attractor_position[jj*2:(jj*2)+2], goal)
                attractor_vel = np.zeros((num_attractor, dim))
                for attractor in range(num_attractor):
                    attractor_vel[attractor, :], state = attractor_dynamic.evaluate(global_attractor_pos[attractor, :], jj)
                # attractor_vel = attractor_dynamic.evaluate(goal.center_position)
                attractor_weights = attractor_dynamic.get_weights_attractors(global_attractor_pos, jj)
                goal_vel, goal_rot = attractor_dynamic.get_goal_velocity(global_attractor_pos, attractor_vel, attractor_weights, jj)

                if state[jj] is False:
                    new_goal_pos = goal_vel * dt_step + goal.center_position
                    new_goal_ori = -goal_rot * dt_step + goal.orientation
                else:
                    new_goal_pos = parking_zone[jj].center_position
                    new_goal_ori = parking_zone[jj].orientation

                # new_goal_pos = attractor_vel * dt_step + goal.center_position
                goal.center_position = new_goal_pos
                goal.orientation = new_goal_ori
                global_attractor_pos = relative2global(relative_attractor_position[jj*2:(jj*2)+2], goal)
                for k in obs_w_multi_agent[jj]:
                    dynamic_avoider.set_attractor_position(global_attractor_pos[k - (jj * 2), :], k)
                # breakpoint()

            for obs in range(num_obs):
                start_time = time.time()
                num_agents_in_obs = len(obs_w_multi_agent[obs])
                # weights = 1 / len(obs_w_multi_agent)
                for agent in obs_w_multi_agent[obs]:
                    # temp_env = obstacle_environment[0:obs] + obstacle_environment[obs + 1:]
                    temp_env = dynamic_avoider.env_slicer(obs)
                    # if (ii % 10) == 0 and ii <= 100:
                    #     attractor_pos = dynamic_avoider.get_attractor_position(agent)
                    #     dynamic_avoider.set_attractor_position(attractor_pos+np.array([0.0, 0.05]), agent)
                    velocity[agent, :] = dynamic_avoider.evaluate_for_crowd_agent(position_list[agent, :, ii - 1],
                                                                                  agent, temp_env)
                    velocity[agent, :] = velocity[agent, :] * weights[obs][agent - (obs * 2)]

                obs_vel = np.zeros(2)
                if obs_w_multi_agent[obs]:
                    for agent in obs_w_multi_agent[obs]:
                        obs_vel += weights[obs][agent - (obs * 2)] * velocity[agent, :]
                else:
                    obs_vel = np.array([-0.3, 0.])

                angular_vel = np.zeros(num_agents_in_obs)
                for agent in obs_w_multi_agent[obs]:
                    angular_vel[agent - (obs * 2)] = weights[obs][agent - (obs * 2)] * np.cross(
                        (obstacle_environment[obs].center_position - position_list[agent, :, ii - 1]),
                        (velocity[agent, :] - obs_vel))

                angular_vel_obs = angular_vel.sum()
                if obs + 1 != num_obs:
                    obstacle_environment[obs].linear_velocity = obs_vel
                    obstacle_environment[obs].angular_velocity = -2 * angular_vel_obs
                    obstacle_environment[obs].do_velocity_step(dt_step)
                else:
                    obstacle_environment[-1].do_velocity_step(dt_step)
                for agent in obs_w_multi_agent[obs]:
                    position_list[agent, :, ii] = obstacle_environment[obs].transform_relative2global(
                        relative_agent_pos[agent, :])

                stop_time = time.time()
                time_list[obs, ii-1] = stop_time-start_time

            for obs in range(num_obs):
                if obs == num_obs - 1:
                    # speed_vec = np.array([0., 0.5])
                    # x_pos = cos(angle) * 2
                    # y_pos = sin(angle) * 2
                    # qolo_dir = angle + pi / 2
                    # obstacle_environment[obs].center_position = np.array([x_pos, y_pos])
                    # obstacle_environment[obs].linear_velocity = speed_vec
                    u_obs_vel = obs_vel / np.linalg.norm(obs_vel)
                    x_vec = np.array([1, 0])
                    dot_prod = np.dot(x_vec, u_obs_vel)
                    qolo_dir = np.arccos(dot_prod)
                    self.update_state_publisher(obs_name[obs], obstacle_environment[-1].center_position, qolo_dir)
                    # angle += degree / 4
                else:
                    self.update_state_publisher(obs_name[obs], obstacle_environment[obs].center_position,
                                                obstacle_environment[obs].orientation)
                    # pass

            loop_rate.sleep()

            # Clear right before drawing again
            ax.clear()

            # Drawing and adjusting of the axis
            for agent in range(num_agent):
                plt.plot(position_list[agent, 0, :ii], position_list[agent, 1, :ii], ':',
                         color='#135e08')
                plt.plot(position_list[agent, 0, ii], position_list[agent, 1, ii],
                         'o', color='#135e08', markersize=12, )
                plt.arrow(position_list[agent, 0, ii], position_list[agent, 1, ii], velocity[agent, 0],
                          velocity[agent, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            for agent in range(num_agent):
                ax.plot(initial_dynamics[agent].attractor_position[0],
                        initial_dynamics[agent].attractor_position[1], 'k*', markersize=8, )
            ax.grid()

            ax.set_aspect('equal', adjustable='box')
            # breakpoiont()

            # Check convergence
            # if np.sum(np.abs(velocity)) < 1e-2:
            #     print(f"Converged at it={ii}")
            #     break

            # print("i am before the sleep")
            plt.pause(dt_sleep)
            # print("i passed the sleep")
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break


def main():
    tot_num_agent = 10
    num_agent = 2
    str_axis = "0.6,0.5".split(",")

    axis = [float(str_axis[0]), float(str_axis[1])]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    rel_agent_pos, radius = calculate_relative_position(num_agent, max_ax_len, min_ax_len)
    rel_agent_pos_table, radius_table = calculate_relative_position(num_agent, 1.6, 0.7)
    obstacle_pos = np.array([[-1.5, 1.2], [-1.5, -1.2], [0, 1.6], [0, -0.8], [1.5, 0.8], [1.5, -1.6], [4.5, -1.2]])
    obstacle_pos = np.array([[-1.5, 1.5], [-1.5, -1.5], [0, 1.5], [0, -1.5], [1.5, 1.5], [1.5, -1.5], [4.5, -1.2]])
    obstacle_pos = np.array([[1, -0.5], [-1, -0.5], [1., 0.5], [-1, 0.5], [0, 0], [2.5, -.4]])
    agent_pos = np.zeros((tot_num_agent, 2))

    attractor_pos = np.zeros((tot_num_agent, 2))

    tot_rel_agent_pos = rel_agent_pos
    for j in range(int((tot_num_agent-4)/2)):
        tot_rel_agent_pos = np.append(tot_rel_agent_pos, rel_agent_pos, axis=0)

    tot_rel_agent_pos = np.append(tot_rel_agent_pos, rel_agent_pos_table, axis=0)

    obstacle_environment = ObstacleContainer()
    # for i in range(len(obstacle_pos)-2):
    #     obstacle_environment.append(
    #         Cuboid(
    #             axes_length=[max_ax_len, min_ax_len],
    #             center_position=obstacle_pos[i],
    #             margin_absolut=radius_table,
    #             orientation=math.pi / 2,
    #             tail_effect=False,
    #             repulsion_coeff=1,
    #         )
    #     )
    obstacle_environment.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[0],
            margin_absolut=radius_table / 1.2,
            orientation=math.pi,
            tail_effect=False,
            repulsion_coeff=1,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[1],
            margin_absolut=radius_table / 1.2,
            orientation=0.,
            tail_effect=False,
            repulsion_coeff=1,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[2],
            margin_absolut=radius_table / 1.2,
            orientation=math.pi,
            tail_effect=False,
            repulsion_coeff=1,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[3],
            margin_absolut=radius_table / 1.2,
            orientation=0.,
            tail_effect=False,
            repulsion_coeff=1,
        )
    )
    obstacle_environment.append(Cuboid(
        axes_length=[1.6, 0.7],
        center_position=obstacle_pos[-2],
        margin_absolut=radius,
        orientation=math.pi / 2,
        tail_effect=False,
        repulsion_coeff=1,
    ))
    obstacle_environment.append(Ellipse(
        axes_length=[0.5, 0.5],
        center_position=obstacle_pos[-1],
        margin_absolut=radius_table,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([-0.3, 0.0]),
    ))

    for i in range(len(obstacle_pos)-2):
        agent_pos[(i*2):(i*2)+2] = relative2global(rel_agent_pos, obstacle_environment[i])

    agent_pos[tot_num_agent-2:tot_num_agent, :] = relative2global(rel_agent_pos_table, obstacle_environment[-2])

    attractor_env = ObstacleContainer()
    # for i in range(len(obstacle_pos)-2):
    #     attractor_env.append(
    #         Cuboid(
    #             axes_length=[max_ax_len, min_ax_len],
    #             center_position=obstacle_pos[i],
    #             margin_absolut=0.,
    #             orientation=math.pi / 2,
    #             tail_effect=False,
    #             repulsion_coeff=1,
    #             linear_velocity=np.array([0., 0.]),
    #         )
    #     )
    attractor_env.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[0],
            margin_absolut=0.,
            orientation=math.pi,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([0., 0.]),
        )
    )
    attractor_env.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[1],
            margin_absolut=0.,
            orientation=0.,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([0., 0.]),
        )
    )
    attractor_env.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[2],
            margin_absolut=0.,
            orientation=math.pi,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([0., 0.]),
        )
    )
    attractor_env.append(
        Cuboid(
            axes_length=[max_ax_len, min_ax_len],
            center_position=obstacle_pos[3],
            margin_absolut=0.,
            orientation=0.,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([0., 0.]),
        )
    )
    attractor_env.append(
        Cuboid(
            axes_length=[1.6, 0.7],
            center_position=obstacle_pos[-2],
            margin_absolut=0.,
            orientation=math.pi / 2,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([0., 0.]),
        )
    )

    for i in range(len(obstacle_pos)-2):
        attractor_pos[(i*2):(i*2)+2] = relative2global(rel_agent_pos, attractor_env[i])

    attractor_pos[tot_num_agent-2:tot_num_agent] = relative2global(rel_agent_pos_table, attractor_env[-2])

    initial_dynamics = []
    for i in range(tot_num_agent):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1, distance_decrease=0.3
            )
        )

    obs_multi_agent = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 5: []}

    DynamicalSystemRviz().run(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        tot_rel_agent_pos,
        attractor_env,
        True,
        x_lim=[-5, 5],
        y_lim=[-4, 4],
        dt_step=0.03,
        dt_sleep=0.01,
    )


if __name__ == "__main__":
    plt.close('all')
    # plt.ion()

    try:
        main()
    except RuntimeError:
        pass
        try:
            rclpy.shutdown()
        except rclpy.RCLError:
            pass
