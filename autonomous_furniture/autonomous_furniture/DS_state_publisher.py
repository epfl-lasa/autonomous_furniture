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


class DynamicalSystemRviz(Node):
    def __init__(self):
        print("line 30")
        rclpy.init()
        super().__init__('DS_state_publisher')
        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()
        self.odom_trans = TransformStamped()
        self.odom_trans.header.frame_id = "odom"

    def __del__(self):
        rclpy.shutdown()

    def update_state_publisher(self, prefix, position, rotation):
        self.odom_trans.child_frame_id = prefix + "base_link"
        now = self.get_clock().now()
        self.odom_trans.header.stamp = now.to_msg()
        self.odom_trans.transform.translation.x = position[0]
        self.odom_trans.transform.translation.y = position[1]
        self.odom_trans.transform.translation.z = 0.
        if "h_bed" in prefix:
            self.odom_trans.transform.rotation = \
                euler_to_quaternion(math.pi/2, 0, rotation)  # rpy
        elif "qolo" in prefix:
            self.odom_trans.transform.rotation = \
                euler_to_quaternion(0, 0, math.pi)  # rpy
        else:
            self.odom_trans.transform.rotation = \
                euler_to_quaternion(0, 0, rotation)  # rpy

        # send the joint state and transform
        self.broadcaster.sendTransform(self.odom_trans)

    def run(
            self, initial_dynamics, obstacle_environment,
            obs_w_multi_agent,
            start_position=None,
            relative_attractor_position=None,
            goal=None,
            x_lim=None, y_lim=None,
            it_max=1000, dt_step=0.03, dt_sleep=0.1
    ):
        loop_rate = self.create_rate(10)

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
        if goal is None:
            goal = Cuboid(
                axes_length=[0.6, 0.6],
                center_position=np.array([0., 0.]),
                margin_absolut=0,
                orientation=0,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0., 0.]),
            )

        attractor_dynamic = AttractorDynamics(obstacle_environment[-1], cutoff_dist=2)
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

        obs_name = ["h_bed_1_", "h_bed_2_", "qolo_"]
        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots(figsize=(10, 8))  # figsize=(10, 8)
        ax.set_aspect(1.0)

        print(f"init dyn: {initial_dynamics[0].attractor_position}")
        print(f"Class dyn avoider: {dynamic_avoider}")

        ii = 0
        while ii < it_max:
            rclpy.spin_once(self)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break

            ii += 1
            if ii > it_max:
                break

            # Here come the main calculation part
            weights = dynamic_avoider.get_influence_weight_at_ctl_points(position_list[:, :, ii-1])
            print(f"weights: {weights}")

            if ii % 10 == 0:
                for jj, obs in enumerate(goal):
                    print(f"     goal: {obs}")              # make a class instead of the dictionary
                    attractor_vel = attractor_dynamic.evaluate(obs.center_position)
                    new_goal_pos = attractor_vel * dt_step + obs.center_position
                    obs.center_position = new_goal_pos
                    global_attractor_pos = relative2global(relative_attractor_position, obs)
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
                    print(f"obs num: {obs}")
                    print(f"ctl num: {agent}")
                    print(f"current ctl pos: {position_list[agent, :, ii - 1]}")
                    print(f"the env: {temp_env}")
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
                    obs_vel = np.array([-0.3, 0.0])

                angular_vel = np.zeros(num_agents_in_obs)
                for agent in obs_w_multi_agent[obs]:
                    angular_vel[agent - (obs * 2)] = weights[obs][agent - (obs * 2)] * np.cross(
                        (obstacle_environment[obs].center_position - position_list[agent, :, ii - 1]),
                        (velocity[agent, :] - obs_vel))

                angular_vel_obs = angular_vel.sum()
                obstacle_environment[obs].linear_velocity = obs_vel
                obstacle_environment[obs].angular_velocity = -angular_vel_obs
                obstacle_environment[obs].do_velocity_step(dt_step)
                for agent in obs_w_multi_agent[obs]:
                    position_list[agent, :, ii] = obstacle_environment[obs].transform_relative2global(
                        relative_agent_pos[agent, :])

                stop_time = time.time()
                time_list[obs, ii-1] = stop_time-start_time

            print(f"Max time: {max(time_list[0, :])}, mean time: {sum(time_list[0, :])/ii}, for obs: {0}, with {len(obs_w_multi_agent[0])} control points")

            for obs in range(num_obs):
                self.update_state_publisher(obs_name[obs], obstacle_environment[obs].center_position, obstacle_environment[obs].orientation)

            print("i am before the loop")
            loop_rate.sleep()
            print("i passed the loop")

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
    center_point = 0.
    tot_num_agent = 4
    num_agent = 2
    str_axis = "2.2,1.1".split(",")

    axis = [float(str_axis[0]), float(str_axis[1])]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    div = max_ax_len / (num_agent + 1)
    # radius = math.sqrt(((min_ax_len / 2) ** 2) + (div ** 2))

    rel_agent_pos, radius = calculate_relative_position(num_agent, max_ax_len, min_ax_len)
    obstacle_pos = np.array([[center_point, 1.5], [0, -1.5], [3.0, -0.2]])
    agent_pos = np.zeros((tot_num_agent, 2))

    for i in range(2):
        agent_pos[i, 1] = 1.5 + ((div * (i+1)) - (max_ax_len / 2))
    for i in [2, 3]:
        agent_pos[i, 1] = -1.5 + ((div * (i+1)) - (max_ax_len / 2))

    attractor_pos = np.zeros((tot_num_agent, 2))
    for i in [0, 1]:
        # attractor_pos[i, 0] = 1.0
        attractor_pos[i, 1] = 1.5 + (div * (i+1)) - (max_ax_len / 2)
    for i in [2, 3]:
        attractor_pos[i, 1] = -1.5 + (div * (i+1)) - (max_ax_len / 2)

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(Cuboid(
        axes_length=[max_ax_len, min_ax_len],
        center_position=obstacle_pos[0],
        margin_absolut=0,
        orientation=math.pi/2,
        tail_effect=False,
        repulsion_coeff=1,
    ))
    obstacle_environment.append(Cuboid(
        axes_length=[max_ax_len, min_ax_len],
        center_position=obstacle_pos[1],  # obstacle_pos[1]  np.array([-1, -2])
        margin_absolut=0,
        orientation=math.pi/2,
        tail_effect=False,
        repulsion_coeff=1,
    ))
    obstacle_environment.append(Ellipse(
        axes_length=[0.6, 0.6],
        center_position=obstacle_pos[2],
        margin_absolut=radius,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([-0.3, 0.0]),
    ))

    agent_pos[:2, :] = relative2global(rel_agent_pos, obstacle_environment[0])
    agent_pos[2:4, :] = relative2global(rel_agent_pos, obstacle_environment[1])

    attractor_env = ObstacleContainer()
    attractor_env.append(Cuboid(
        axes_length=[max_ax_len, min_ax_len],
        center_position=obstacle_pos[0],
        margin_absolut=0.,
        orientation=math.pi / 2,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([0., 0.]),
    ))
    attractor_env.append(Cuboid(
        axes_length=[max_ax_len, min_ax_len],
        center_position=obstacle_pos[1],
        margin_absolut=0.,
        orientation=math.pi / 2,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([0., 0.]),
    ))

    attractor_pos[:2, :] = relative2global(rel_agent_pos, attractor_env[0])
    attractor_pos[2:4, :] = relative2global(rel_agent_pos, attractor_env[1])

    initial_dynamics = [
        LinearSystem(
            attractor_position=attractor_pos[0],
            maximum_velocity=1, distance_decrease=0.3
        ),
        LinearSystem(
            attractor_position=attractor_pos[1],
            maximum_velocity=1, distance_decrease=0.3
        ),
    ]

    initial_dynamics = []
    for i in range(tot_num_agent):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1, distance_decrease=0.3
            )
        )

    # obs_multi_agent = {0: [0, 1], 1: []}
    # obs_multi_agent = {0: [0, 1], 1: [], 2: []}
    obs_multi_agent = {0: [0, 1], 1: [2, 3], 2: []}
    # for i in range(num_agent):
    #     obs_multi_agent[0].append(i)

    DynamicalSystemRviz().run(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        rel_agent_pos,
        attractor_env,
        x_lim=[-4, 3],
        y_lim=[-3, 3],
        dt_step=0.05,
    )


if __name__ == "__main__":
    plt.close('all')
    # plt.ion()

    try:
        main()
    except RuntimeError:
        try:
            rclpy.shutdown()
        except RCLError:
            pass
