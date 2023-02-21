import math
from math import sin, cos, pi
import time
import numpy as np
import matplotlib.pyplot as plt

# from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
# from dynamic_obstacle_avoidance.obstacles import Cuboid,
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from autonomous_furniture.agent import Furniture, Person

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.avoidance.dynamic_crowd_avoider import (
    obstacle_environment_slicer,
)

from vartools.dynamical_systems import LinearSystem

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformBroadcaster, TransformStamped

from autonomous_furniture.analysis.calc_time import (
    calculate_relative_position,
    relative2global,
    global2relative,
)
from autonomous_furniture.attractor_dynamics import AttractorDynamics


def euler_to_quaternion(roll, pitch, yaw):
    # TODO: this rotation should be replaced with scipy-Rotations (!)
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


class DynamicalSystemRviz(Node):
    def __init__(self):
        # self.animation_paused = True
        self.animation_paused = False

        rclpy.init()

        super().__init__("furniture_publisher")

        qos_profile = QoSProfile(depth=10)

        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()
        self.odom_trans = TransformStamped()
        self.odom_trans.header.frame_id = "world"

    def __del__(self):
        rclpy.shutdown()

    def on_click(self, event):
        self.animation_paused = not self.animation_paused

    def update_state_publisher(self, prefix, position, rotation):
        self.odom_trans.child_frame_id = prefix + "/" + "base_link"
        # print("Doing", self.odom_trans.child_frame_id)

        now = self.get_clock().now()
        self.odom_trans.header.stamp = now.to_msg()
        self.odom_trans.transform.translation.x = position[0]
        self.odom_trans.transform.translation.y = position[1]
        self.odom_trans.transform.translation.z = 0.0
        self.odom_trans.transform.rotation = euler_to_quaternion(0, 0, rotation)  # rpy

        if "table" in prefix:
            self.odom_trans.transform.translation.x = position[0] + 0.2

        if "qolo" in prefix:
            self.odom_trans.transform.translation.z = 0.2
            self.odom_trans.transform.rotation = euler_to_quaternion(
                0, 0, rotation
            )  # rpy

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
        dt_sleep=0.1,
    ):
        loop_rate = self.create_rate(30)

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
            relative_attractor_position = np.array([0.0, 0.0])
        if goals is None:
            goals = Cuboid(
                axes_length=[0.6, 0.6],
                center_position=np.array([0.0, 0.0]),
                margin_absolut=0,
                orientation=0,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0.0, 0.0]),
            )
        if walls is True:
            walls_center_position = np.array(
                [[0.0, y_lim[0]], [x_lim[0], 0.0], [0.0, y_lim[1]], [x_lim[1], 0.0]]
            )
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
                    linear_velocity=np.array([0.0, 0.0]),
                    is_boundary=False,
                ),
                Cuboid(
                    axes_length=[0.1, y_length],
                    center_position=walls_center_position[1],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0.0, 0.0]),
                ),
                Cuboid(
                    axes_length=[x_length, 0.1],
                    center_position=walls_center_position[2],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0.0, 0.0]),
                ),
                Cuboid(
                    axes_length=[0.1, y_length],
                    center_position=walls_center_position[3],
                    margin_absolut=wall_margin,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0.0, 0.0]),
                ),
            ]
            obstacle_environment = (
                obstacle_environment[:-1] + walls_cont + obstacle_environment[-1:]
            )
        else:
            wall_margin = 0.0

        x_offset = 1.5
        y_offset = 1.0
        parking_zone_cp = np.array(
            [
                [1.0 + x_offset, -0.42 + y_offset],
                [-1.0 + x_offset, -0.42 + y_offset],
                [1.0 + x_offset, 0.42 + y_offset],
                [-1 + x_offset, 0.42 + y_offset],
                [0.0 + x_offset, 0.0 + y_offset],
            ]
        )
        parking_zone_or = [pi, 0.0, pi, 0.0, pi / 2]
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
                    linear_velocity=np.array([0.0, 0.0]),
                )
            )

        attractor_dynamic = AttractorDynamics(
            obstacle_environment, cutoff_dist=1.8, parking_zone=parking_zone
        )
        dynamic_avoider = DynamicCrowdAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_environment,
            obs_multi_agent=obs_w_multi_agent,
        )
        position_list = np.zeros((num_agent, dim, it_max))
        time_list = np.zeros((num_obs, it_max))
        relative_agent_pos = np.zeros((num_agent, dim))

        for obs in range(num_obs):
            relative_agent_pos[obs_w_multi_agent[obs], :] = global2relative(
                start_position[obs_w_multi_agent[obs]], obstacle_environment[obs]
            )

        n_chairs = 4
        obs_name = ["chair" + str(ii) for ii in range(n_chairs)]
        obs_name = obs_name + ["table", "qolo_human"]
        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots()  # figsize=(10, 8)
        cid = fig.canvas.mpl_connect("button_press_event", self.on_click)
        ax.set_aspect(1.0)

        print("Starting the simulation....")
        ii = 0
        while ii < it_max:
            if not ii % 10:
                print(f"Doing it={ii} / {it_max} ")
            rclpy.spin_once(self)

            if self.animation_paused:
                print("Taking a pause.")
                plt.pause(dt_sleep)
                if not plt.fignum_exists(fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                continue

            ii += 1
            if ii > it_max:
                break

            # Here come the main calculation part
            weights = dynamic_avoider.get_influence_weight_at_ctl_points(
                position_list[:, :, ii - 1], 3
            )

            for jj, goal in enumerate(goals):
                num_attractor = len(obs_w_multi_agent[jj])
                global_attractor_pos = relative2global(
                    relative_attractor_position[jj * 2 : (jj * 2) + 2], goal
                )
                attractor_vel = np.zeros((num_attractor, dim))
                for attractor in range(num_attractor):
                    attractor_vel[attractor, :], state = attractor_dynamic.evaluate(
                        global_attractor_pos[attractor, :], jj
                    )
                attractor_weights = attractor_dynamic.get_weights_attractors(
                    global_attractor_pos, jj
                )
                goal_vel, goal_rot = attractor_dynamic.get_goal_velocity(
                    global_attractor_pos, attractor_vel, attractor_weights, jj
                )

                if state[jj] is False:
                    new_goal_pos = goal_vel * dt_step + goal.center_position
                    new_goal_ori = -goal_rot * dt_step + goal.orientation
                else:
                    new_goal_pos = parking_zone[jj].center_position
                    new_goal_ori = parking_zone[jj].orientation

                goal.center_position = new_goal_pos
                goal.orientation = new_goal_ori
                global_attractor_pos = relative2global(
                    relative_attractor_position[jj * 2 : (jj * 2) + 2], goal
                )
                for k in obs_w_multi_agent[jj]:
                    dynamic_avoider.set_attractor_position(
                        global_attractor_pos[k - (jj * 2), :], k
                    )

            for obs in range(num_obs):
                start_time = time.time()
                num_agents_in_obs = len(obs_w_multi_agent[obs])
                for agent in obs_w_multi_agent[obs]:
                    # temp_env = dynamic_avoider.env_slicer(obs)
                    temp_env = obstacle_environment_slicer(
                        dynamic_avoider.obstacle_environment, obs_index=obs
                    )
                    velocity[agent, :] = dynamic_avoider.evaluate_for_crowd_agent(
                        position_list[agent, :, ii - 1], agent, temp_env
                    )
                    velocity[agent, :] = (
                        velocity[agent, :] * weights[obs][agent - (obs * 2)]
                    )

                obs_vel = np.zeros(2)
                if obs_w_multi_agent[obs]:
                    for agent in obs_w_multi_agent[obs]:
                        obs_vel += weights[obs][agent - (obs * 2)] * velocity[agent, :]
                else:
                    obs_vel = np.array([-0.3, 0.0])

                angular_vel = np.zeros(num_agents_in_obs)
                for agent in obs_w_multi_agent[obs]:
                    angular_vel[agent - (obs * 2)] = weights[obs][
                        agent - (obs * 2)
                    ] * np.cross(
                        (
                            obstacle_environment[obs].center_position
                            - position_list[agent, :, ii - 1]
                        ),
                        (velocity[agent, :] - obs_vel),
                    )

                angular_vel_obs = angular_vel.sum()
                if obs + 1 != num_obs:
                    obstacle_environment[obs].linear_velocity = obs_vel
                    obstacle_environment[obs].angular_velocity = -2 * angular_vel_obs
                    obstacle_environment[obs].do_velocity_step(dt_step)
                else:
                    obstacle_environment[-1].do_velocity_step(dt_step)
                for agent in obs_w_multi_agent[obs]:
                    position_list[agent, :, ii] = obstacle_environment[
                        obs
                    ].transform_relative2global(relative_agent_pos[agent, :])

                stop_time = time.time()
                time_list[obs, ii - 1] = stop_time - start_time

            for obs in range(num_obs):
                if obs == num_obs - 1:
                    u_obs_vel = obs_vel / np.linalg.norm(obs_vel)
                    x_vec = np.array([1, 0])
                    dot_prod = np.dot(x_vec, u_obs_vel)
                    qolo_dir = np.arccos(dot_prod)
                    self.update_state_publisher(
                        obs_name[obs],
                        obstacle_environment[-1].center_position,
                        qolo_dir,
                    )
                else:
                    self.update_state_publisher(
                        obs_name[obs],
                        obstacle_environment[obs].center_position,
                        obstacle_environment[obs].orientation,
                    )

            loop_rate.sleep()

            # Clear right before drawing again
            ax.clear()

            # Drawing and adjusting of the axis
            for agent in range(num_agent):
                plt.plot(
                    position_list[agent, 0, :ii],
                    position_list[agent, 1, :ii],
                    ":",
                    color="#135e08",
                )
                plt.plot(
                    position_list[agent, 0, ii],
                    position_list[agent, 1, ii],
                    "o",
                    color="#135e08",
                    markersize=12,
                )
                plt.arrow(
                    position_list[agent, 0, ii],
                    position_list[agent, 1, ii],
                    velocity[agent, 0],
                    velocity[agent, 1],
                    head_width=0.05,
                    head_length=0.1,
                    fc="k",
                    ec="k",
                )

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            plot_obstacles(
                ax=ax,
                obstacle_container=obstacle_environment,
                x_lim=x_lim,
                y_lim=y_lim,
                showLabel=False,
            )

            for agent in range(num_agent):
                ax.plot(
                    initial_dynamics[agent].attractor_position[0],
                    initial_dynamics[agent].attractor_position[1],
                    "k*",
                    markersize=8,
                )
            ax.grid()

            ax.set_aspect("equal", adjustable="box")

            plt.pause(dt_sleep)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break


def main():
    tot_num_agent = 10
    num_agent = 2

    axis = [0.6, 0.5]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    rel_agent_pos, radius = calculate_relative_position(
        num_agent, max_ax_len, min_ax_len
    )
    rel_agent_pos_table, radius_table = calculate_relative_position(num_agent, 1.6, 0.7)
    chair_pos = np.array(
        [[1, -0.42], [-1, -0.42], [1.0, 0.42], [-1, 0.42]]
    )
    table_pos = np.array([[0,0]])
    human_pos = np.array([[3.0, -0.2]])
    shift_vectors = np.array([[-2,-2]]) #shift the positions around in the room
    furn_tot = (len(table_pos)+len(human_pos)+len(chair_pos))*len(shift_vectors)
    
    agent_pos = np.zeros((tot_num_agent, 2))

    attractor_pos = np.zeros((tot_num_agent, 2))

    tot_rel_agent_pos = rel_agent_pos
    for j in range(int((tot_num_agent - 4) / 2)):
        tot_rel_agent_pos = np.append(tot_rel_agent_pos, rel_agent_pos, axis=0)

    tot_rel_agent_pos = np.append(tot_rel_agent_pos, rel_agent_pos_table, axis=0)

    chair_margin = 0.2
    chair_orientation = 0.0
    obstacle_environment = ObstacleContainer()

    for i in range(len(shift_vectors)):
        #add chairs
        for j in range(len(chair_pos)):
            obstacle_environment.append(
                Cuboid(
                    axes_length=[max_ax_len, min_ax_len],
                    center_position=chair_pos[j]+shift_vectors[i],
                    margin_absolut=chair_margin,
                    orientation=chair_orientation,
                    tail_effect=False,
                    repulsion_coeff=1,
                )
            )
        #add tables
        for j in range(len(table_pos)):
            obstacle_environment.append(
                Cuboid(
                    axes_length=[1.6, 0.7],
                    center_position=table_pos[j]+shift_vectors[i],
                    margin_absolut=radius / 1.1,
                    orientation=math.pi / 2,
                    tail_effect=False,
                    repulsion_coeff=1,
                )
            )
        #add person
        for j in range(len(human_pos)):
            obstacle_environment.append(
                Ellipse(
                    axes_length=[0.5, 0.5],
                    center_position=human_pos[j]+shift_vectors[i],
                    margin_absolut=radius_table,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([-0.3, 0.3]),
                )
            )
    

    for i in range(furn_tot - 2):
        agent_pos[(i * 2) : (i * 2) + 2] = relative2global(
            rel_agent_pos, obstacle_environment[i]
        )

    agent_pos[tot_num_agent - 2 : tot_num_agent, :] = relative2global(
        rel_agent_pos_table, obstacle_environment[-2]
    )

    attractor_env = ObstacleContainer()
    for i in range(len(shift_vectors)):
        #add chairs
        for j in range(len(chair_pos)):
            attractor_env.append(
                Cuboid(
                    axes_length=[max_ax_len, min_ax_len],
                    center_position=chair_pos[j]+shift_vectors[i],
                    margin_absolut=chair_margin,
                    orientation=chair_orientation,
                    tail_effect=False,
                    repulsion_coeff=1,
                )
            )
        #add tables
        for j in range(len(table_pos)):
            attractor_env.append(
                Cuboid(
                    axes_length=[1.6, 0.7],
                    center_position=table_pos[j]+shift_vectors[i],
                    margin_absolut=radius / 1.1,
                    orientation=math.pi / 2,
                    tail_effect=False,
                    repulsion_coeff=1,
                )
            )
        #add person
        for j in range(len(human_pos)):
            attractor_env.append(
                Ellipse(
                    axes_length=[0.5, 0.5],
                    center_position=human_pos[j]+shift_vectors[i],
                    margin_absolut=radius_table,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([-0.3, 0.3]),
                )
            )
        
    for i in range(furn_tot - 2):
        attractor_pos[(i * 2) : (i * 2) + 2] = relative2global(
            rel_agent_pos, attractor_env[i]
        )

    attractor_pos[tot_num_agent - 2 : tot_num_agent] = relative2global(
        rel_agent_pos_table, attractor_env[-2]
    )

    initial_dynamics = []
    for i in range(tot_num_agent):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1,
                distance_decrease=0.3,
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
        x_lim=[-4.5, 4.5],
        y_lim=[-3.5, 3.5],
        dt_step=0.03,
        dt_sleep=0.01,
    )


if __name__ == "__main__":
    plt.close("all")

    # try:
    main()
    # except RuntimeError:
    # rclpy.shutdown()
