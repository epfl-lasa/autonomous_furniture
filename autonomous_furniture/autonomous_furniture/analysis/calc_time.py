import math
from math import cos, sin
import time
import numpy as np
import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from autonomous_furniture.attractor_dynamics import AttractorDynamics
from autonomous_furniture.furniture_class import Furniture
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_ctl",
    action="store",
    default=2,
    help="int of number of control points in the furniture",
)
parser.add_argument(
    "--rect_size",
    action="store",
    default="2.2,1.1",
    help="x,y of the max size of the furniture",
)
args = parser.parse_args()


class DynamicalSystemAnimation:
    def __init__(self):
        self.animation_paused = False

    def on_click(self, event):
        self.animation_paused = not self.animation_paused

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
        it_max=1100,
        dt_step=0.03,
        dt_sleep=0.1,
    ):
        num_obs = len(obstacle_environment)
        if start_position.ndim > 1:
            num_agent = len(start_position)
        else:
            num_agent = 1
        dim = 2

        obs_velocities = np.zeros((num_obs, it_max, dim))
        obs_rot = np.zeros((it_max, num_obs))
        obs_positions = np.zeros((num_obs, it_max, dim))
        agent_dist = np.zeros((it_max, num_agent))
        agent_dist_all = np.zeros((it_max, num_agent))
        agent_speed = np.zeros((num_agent, it_max, dim))
        agent_weights = np.zeros((it_max, num_agent))
        attractor_velocities = np.zeros((num_agent, it_max, dim))
        attract_weights = np.zeros((it_max, num_agent))
        attractor_dist = np.zeros((it_max, num_agent))

        if y_lim is None:
            y_lim = [-3.0, 3.0]
        if x_lim is None:
            x_lim = [-3.0, 3.0]
        if start_position is None:
            start_position = np.zeros((num_obs, dim))
        if num_agent > 1:
            velocity = np.zeros((num_agent, dim))
        else:
            velocity = np.zeros((2, dim))
        if relative_attractor_position is None:
            relative_attractor_position = np.array([0.0, 0.0])
        if goals is None:
            goals = ObstacleContainer()
            goals.append(
                Cuboid(
                    axes_length=[0.6, 0.6],
                    center_position=np.array([0.0, 0.0]),
                    margin_absolut=0,
                    orientation=0,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0.0, 0.0]),
                )
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

        parking_zone = np.array(
            [
                [x_lim[0] + wall_margin, y_lim[1] - wall_margin],
                [x_lim[1] - wall_margin, y_lim[0] + wall_margin],
            ]
        )
        max_axis = max(goals[0].axes_length)
        min_axis = min(goals[0].axes_length)
        offset = 0.3
        wall_thickness = 0.11
        # parking_zone_cp = np.array([[x_lim[0] + (wall_margin + wall_thickness), y_lim[0] + (offset + max_axis/2)],
        #                             [x_lim[0] + (wall_margin + wall_thickness), y_lim[1] - (offset + max_axis/2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness), y_lim[1] - (offset + max_axis/2)],
        #                             [x_lim[1] - (wall_margin + wall_thickness), y_lim[0] + (offset + max_axis/2)]])
        parking_zone_cp = np.array(
            [
                [
                    x_lim[1] - (wall_margin + wall_thickness),
                    y_lim[0] + (offset + max_axis / 2),
                ],
                [
                    x_lim[1] - (wall_margin + wall_thickness + min_axis + offset),
                    y_lim[0] + (offset + max_axis / 2),
                ],
                [
                    x_lim[1] - (wall_margin + wall_thickness + 2 * (min_axis + offset)),
                    y_lim[0] + (offset + max_axis / 2),
                ],
                [
                    x_lim[1] - (wall_margin + wall_thickness + 3 * (min_axis + offset)),
                    y_lim[0] + (offset + max_axis / 2),
                ],
            ]
        )
        parking_zone = ObstacleContainer()
        for pk in range(len(parking_zone_cp)):
            parking_zone.append(
                Cuboid(
                    axes_length=goals[pk].axes_length,
                    center_position=parking_zone_cp[pk],
                    margin_absolut=0,
                    orientation=math.pi / 2,
                    tail_effect=False,
                    repulsion_coeff=1,
                    linear_velocity=np.array([0.0, 0.0]),
                )
            )

        attractor_dynamic = AttractorDynamics(
            obstacle_environment, cutoff_dist=2, parking_zone=parking_zone
        )
        dynamic_avoider = DynamicCrowdAvoider(
            initial_dynamics=initial_dynamics,
            environment=obstacle_environment,
            obs_multi_agent=obs_w_multi_agent,
        )
        position_list = np.zeros((num_agent, dim, it_max))
        time_list = np.zeros((num_obs, it_max))
        relative_agent_pos = np.zeros((num_agent, dim))

        only_person_container = ObstacleContainer()
        only_person_container.append(obstacle_environment[-1])

        for obs in range(num_obs):
            relative_agent_pos[obs_w_multi_agent[obs], :] = global2relative(
                start_position[obs_w_multi_agent[obs]], obstacle_environment[obs]
            )

        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots(figsize=(10, 8))  # figsize=(10, 8)
        ax.set_aspect(1.0)
        cid = fig.canvas.mpl_connect("button_press_event", self.on_click)

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

            # Here come the main calculation part
            (
                weights,
                agent_dist_all[ii - 1, :],
                agent_weights[ii - 1, :],
            ) = dynamic_avoider.get_influence_weight_at_ctl_points(
                position_list[:, :, ii - 1], 3, True
            )
            agent_dist[ii - 1, :] = dynamic_avoider.get_gamma_at_pts(
                position_list[:, :, ii - 1], only_person_container
            )

            prev_num_attractor = 0
            for jj, goal in enumerate(goals):
                att_start_time = time.time()
                num_attractor = len(obs_w_multi_agent[jj])
                global_attractor_pos = relative2global(
                    relative_attractor_position, goal
                )
                attractor_vel = np.zeros((num_attractor, dim))
                for attractor in range(num_attractor):
                    attractor_vel[attractor, :], state = attractor_dynamic.evaluate(
                        global_attractor_pos[attractor, :], jj
                    )
                # attractor_vel = attractor_dynamic.evaluate(obs.center_position)
                attractor_weights = attractor_dynamic.get_weights_attractors(
                    global_attractor_pos, jj
                )
                goal_vel, goal_rot = attractor_dynamic.get_goal_velocity(
                    global_attractor_pos, attractor_vel, attractor_weights, jj
                )

                temp_vel = attractor_weights * attractor_vel
                attractor_velocities[
                    prev_num_attractor : prev_num_attractor + num_attractor, ii - 1, :
                ] = temp_vel
                attract_weights[
                    ii - 1, prev_num_attractor : prev_num_attractor + num_attractor
                ] = attractor_weights
                attractor_dist[
                    ii - 1, prev_num_attractor : prev_num_attractor + num_attractor
                ] = attractor_dynamic.get_gamma_at_attractor(
                    global_attractor_pos, only_person_container
                )
                prev_num_attractor += num_attractor

                # print(state)
                if state[jj] is False:
                    new_goal_pos = goal_vel * dt_step + goal.center_position
                    new_goal_ori = -goal_rot * dt_step + goal.orientation
                else:
                    new_goal_pos = parking_zone[3 - jj].center_position
                    new_goal_ori = parking_zone[jj].orientation
                goal.center_position = new_goal_pos
                goal.orientation = new_goal_ori

                # attractor_vel = attractor_dynamic.evaluate(goal.center_position)
                # new_goal_pos = attractor_vel * dt_step + goal.center_position
                # goal.center_position = new_goal_pos
                global_attractor_pos = relative2global(
                    relative_attractor_position, goal
                )
                for i in obs_w_multi_agent[jj]:
                    # initial_dynamics[i].attractor_position = global_attractor_pos[i]
                    dynamic_avoider.set_attractor_position(
                        global_attractor_pos[i - (jj * 2)], i
                    )
                att_stop_time = time.time()
                att_tot_time = att_stop_time - att_start_time
                # print(f"Current time of attractor moving: {att_tot_time} for goal: {jj}")
                # attractor_dynamic.print_state(jj)

            for obs in range(num_obs):
                start_time = time.time()
                num_agents_in_obs = len(obs_w_multi_agent[obs])
                # weights = 1 / len(obs_w_multi_agent)
                for agent in obs_w_multi_agent[obs]:
                    temp_env = dynamic_avoider.env_slicer(obs)
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
                    obs_velocities[obs, ii - 1, :] = obs_vel
                    obs_rot[ii - 1, obs] = -2 * angular_vel_obs
                    obs_positions[obs, ii - 1, :] = obstacle_environment[
                        obs
                    ].center_position
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

                # print(f"Max time: {max(time_list[obs, :])}, mean time: {sum(time_list[obs, :])/ii}, for obs: {obs}, with {len(obs_w_multi_agent[obs])} control points")

            agent_speed[:, ii - 1, :] = velocity

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

                ax.plot(
                    initial_dynamics[agent].attractor_position[0],
                    initial_dynamics[agent].attractor_position[1],
                    "k*",
                    markersize=8,
                )

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            plot_obstacles(
                ax,
                obstacle_environment,
                x_lim,
                y_lim,
                showLabel=False,
                border_linestyle="--",
            )

            # for agent in range(num_agent):
            #     ax.plot(initial_dynamics[agent].attractor_position[0],
            #             initial_dynamics[agent].attractor_position[1], 'k*', markersize=8,)
            ax.grid()

            ax.set_aspect("equal", adjustable="box")
            # breakpoiont()

            # Check convergence
            # if np.sum(np.abs(velocity)) < 1e-2:
            #     print(f"Converged at it={ii}")
            #     break

            plt.pause(dt_sleep / 10)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break

            if ii == it_max - 1:
                source = "data/"
                for obs in range(num_obs):
                    temp_array = np.asarray(obs_velocities[obs, :, :])
                    np.savetxt(
                        source + "obs_vel_" + str(obs) + ".csv",
                        temp_array,
                        delimiter=",",
                    )
                    temp_array = np.asarray(obs_positions[obs, :, :])
                    np.savetxt(
                        source + "obs_pos_" + str(obs) + ".csv",
                        temp_array,
                        delimiter=",",
                    )
                temp_array = np.asarray(obs_rot)
                np.savetxt(source + "obs_rot" + ".csv", temp_array, delimiter=",")
                temp_array = np.asarray(agent_dist)
                np.savetxt(source + "agent_dist" + ".csv", temp_array, delimiter=",")
                temp_array = np.asarray(agent_dist_all)
                np.savetxt(
                    source + "agent_dist_all" + ".csv", temp_array, delimiter=","
                )
                temp_array = np.asarray(agent_weights)
                np.savetxt(source + "agent_weights" + ".csv", temp_array, delimiter=",")
                temp_array = np.asarray(attract_weights)
                np.savetxt(
                    source + "attractor_weights" + ".csv", temp_array, delimiter=","
                )
                temp_array = np.asarray(attractor_dist)
                np.savetxt(
                    source + "attractor_dist" + ".csv", temp_array, delimiter=","
                )
                for agent in range(num_agent):
                    temp_array = np.asarray(agent_speed[agent, :, :])
                    np.savetxt(
                        source + "agent_vel_" + str(agent) + ".csv",
                        temp_array,
                        delimiter=",",
                    )
                    temp_array = np.asarray(attractor_velocities[agent, :, :])
                    np.savetxt(
                        source + "attractor_vel_" + str(agent) + ".csv",
                        temp_array,
                        delimiter=",",
                    )


def calculate_relative_position(num_agent, max_ax, min_ax):
    div = max_ax / (num_agent + 1)
    radius = math.sqrt(((min_ax / 2) ** 2) + (div**2))
    rel_agent_pos = np.zeros((num_agent, 2))

    for i in range(num_agent):
        rel_agent_pos[i, 0] = (div * (i + 1)) - (max_ax / 2)

    return rel_agent_pos, radius


def relative2global(relative_pos, obstacle):
    angle = obstacle.orientation
    obs_pos = obstacle.center_position
    # print(f"obs pos: {obs_pos}")
    global_pos = np.zeros_like(relative_pos)
    # print(f"rel: {relative_pos}")
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    for i in range(relative_pos.shape[0]):
        rot_rel_pos = np.dot(rot, relative_pos[i, :])
        global_pos[i, :] = obs_pos + rot_rel_pos

    # print(f"glob: {global_pos}")
    return global_pos


def global2relative(global_pos, obstacle):
    angle = -1 * obstacle.orientation
    obs_pos = obstacle.center_position
    relative_pos = np.zeros_like(global_pos)
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    for i in range(global_pos.shape[0]):
        rel_pos_pre_rot = global_pos[i, :] - obs_pos
        relative_pos[i, :] = np.dot(rot, rel_pos_pre_rot)

    return relative_pos


def multiple_robots():
    num_agent = int(args.num_ctl)
    str_axis = args.rect_size.split(",")
    axis = [float(str_axis[0]), float(str_axis[1])]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    tot_ctl_pts = 8
    # div = max_ax_len / (num_agent + 1)
    # radius = math.sqrt(((min_ax_len / 2) ** 2) + (div ** 2))
    # obstacle_pos = np.array([[-1.5, 1.5], [-1.5, -1.5], [0, 1.5], [0, -1.5], [1.5, 1.5], [1.5, -1.5], [4.5, -1.2]])
    obstacle_pos = np.array(
        [[-1.5, 1.5], [-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [4.5, -1.2]]
    )
    # obstacle_pos = np.array([[1., 0.], [4, 0.]])
    # agent_pos = np.zeros((num_agent, 2))
    # for i in range(num_agent):
    #     agent_pos[i, 0] = - center_point + ((div * (i+1)) - (max_ax_len / 2))
    # attractor_pos = np.zeros((num_agent, 2))
    # for i in range(num_agent):
    #     attractor_pos[i, 0] = 1.0
    #     attractor_pos[i, 1] = (div * (i+1)) - (max_ax_len / 2)

    rel_agent_pos, radius = calculate_relative_position(
        num_agent, max_ax_len, min_ax_len
    )

    obstacle_environment = ObstacleContainer()
    for i in range(len(obstacle_pos) - 1):
        obstacle_environment.append(
            Cuboid(
                axes_length=[max_ax_len, min_ax_len],
                center_position=obstacle_pos[i],
                margin_absolut=radius / 1.5,
                orientation=math.pi / 2,
                tail_effect=False,
                repulsion_coeff=1,
            )
        )
    # obstacle_environment.append(Cuboid(
    #     axes_length=[max_ax_len, min_ax_len],
    #     center_position=obstacle_pos[0],
    #     margin_absolut=0,
    #     orientation=math.pi/2,
    #     tail_effect=False,
    #     repulsion_coeff=1,
    # ))
    # obstacle_environment.append(Cuboid(
    #     axes_length=[max_ax_len, min_ax_len],
    #     center_position=obstacle_pos[1],
    #     margin_absolut=0,
    #     orientation=math.pi/2,
    #     tail_effect=False,
    #     repulsion_coeff=1,
    # ))
    obstacle_environment.append(
        Ellipse(
            axes_length=[0.6, 0.6],
            center_position=obstacle_pos[-1],
            margin_absolut=radius,
            orientation=0,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=np.array([-0.3, 0.1]),
        )
    )

    agent_pos = np.zeros((tot_ctl_pts, 2))
    for i in range(len(obstacle_pos) - 1):
        agent_pos[(i * 2) : (i * 2) + 2] = relative2global(
            rel_agent_pos, obstacle_environment[i]
        )
    # agent_pos[:2, :] = relative2global(rel_agent_pos, obstacle_environment[0])
    # agent_pos[2:, :] = relative2global(rel_agent_pos, obstacle_environment[1])

    attractor_env = ObstacleContainer()
    for i in range(len(obstacle_pos) - 1):
        attractor_env.append(
            Cuboid(
                axes_length=[max_ax_len, min_ax_len],
                center_position=obstacle_pos[i],
                margin_absolut=0.0,
                orientation=math.pi / 2,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0.0, 0.0]),
            )
        )
    # attractor_env.append(Cuboid(
    #     axes_length=[max_ax_len, min_ax_len],
    #     center_position=obstacle_pos[0],
    #     margin_absolut=0.,
    #     orientation=math.pi/2,
    #     tail_effect=False,
    #     repulsion_coeff=1,
    #     linear_velocity=np.array([0., 0.]),
    # ))
    # attractor_env.append(Cuboid(
    #     axes_length=[max_ax_len, min_ax_len],
    #     center_position=obstacle_pos[1],
    #     margin_absolut=0.,
    #     orientation=math.pi/2,
    #     tail_effect=False,
    #     repulsion_coeff=1,
    #     linear_velocity=np.array([0., 0.]),
    # ))

    attractor_pos = np.zeros((tot_ctl_pts, 2))
    for i in range(len(obstacle_pos) - 1):
        attractor_pos[(i * 2) : (i * 2) + 2] = relative2global(
            rel_agent_pos, attractor_env[i]
        )
    # attractor_pos[:2, :] = relative2global(rel_agent_pos, attractor_env[0])
    # attractor_pos[2:, :] = relative2global(rel_agent_pos, attractor_env[1])

    # initial_dynamics = [LinearSystem(
    #     attractor_position=attractor_pos[0],
    #     maximum_velocity=1, distance_decrease=0.3
    # ),
    #     LinearSystem(
    #         attractor_position=attractor_pos[1],
    #         maximum_velocity=1, distance_decrease=0.3
    #     )
    # ]
    initial_dynamics = []
    for i in range(tot_ctl_pts):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1,
                distance_decrease=0.3,
            )
        )

    # obs_multi_agent = {0: [0, 1], 1: []}
    # obs_multi_agent = {0: [0, 1], 1: [2, 3], 2: []}
    # obs_multi_agent = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 5: [10, 11], 6: []}
    obs_multi_agent = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: []}
    # obs_multi_agent = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 5: [10, 11], 6: [], 7: [], 8: [], 9: [], 10: []}
    # for i in range(num_agent):
    #     obs_multi_agent[0].append(i)

    DynamicalSystemAnimation().run(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        rel_agent_pos,
        attractor_env,
        True,
        x_lim=[-6, 6],
        y_lim=[-5, 5],
        dt_step=0.05,
        dt_sleep=0.01,
    )


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    multiple_robots()
