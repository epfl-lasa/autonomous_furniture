import numpy as np
import math
import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from autonomous_furniture.attractor_dynamics import AttractorDynamics
from autonomous_furniture.furniture_class import Furniture, FurnitureDynamics
from calc_time import calculate_relative_position, relative2global, global2relative


class DynamicFurniture:
    def __init__(self):
        self.animation_paused = True

    def on_click(self, event):
        self.animation_paused = not self.animation_paused

    def run(
            self, furniture_env, initial_dynamics, obstacle_environment,
            obs_w_multi_agent,
            start_position=None,
            relative_attractor_position=None,
            goals=None,
            x_lim=None, y_lim=None,
            it_max=1000, dt_step=0.03, dt_sleep=0.1
    ):

        num_obs = len(furniture_env)
        total_ctl_pts = 0
        for furniture in furniture_env:
            if furniture.is_controllable:
                total_ctl_pts += furniture.num_control_points
        dim = 2

        if y_lim is None:
            y_lim = [-3., 3.]
        if x_lim is None:
            x_lim = [-3., 3.]
        if start_position is None:
            start_position = np.zeros((num_obs, dim))
        if total_ctl_pts > 1:
            velocity = np.zeros((total_ctl_pts, dim))
        else:
            velocity = np.zeros((2, dim))
        if relative_attractor_position is None:
            relative_attractor_position = np.array([0., 0.])
        if goals is None:
            goals = Ellipse(
                axes_length=[0.6, 0.6],
                center_position=np.array([0., 0.]),
                margin_absolut=0,
                orientation=0,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0., 0.]),
            )

        for furniture in furniture_env:
            if not furniture.is_controllable:
                person = furniture.furniture_container

        attractor_dynamic = AttractorDynamics(person, cutoff_dist=3)
        dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=initial_dynamics, environment=obstacle_environment,
                                              obs_multi_agent=obs_w_multi_agent)
        furniture_avoider = FurnitureDynamics(initial_dynamics, furniture_env)
        position_list = np.zeros((total_ctl_pts, dim, it_max))
        relative_agent_pos = np.zeros((total_ctl_pts, dim))

        for obs in range(num_obs):
            if furniture_env[obs].is_controllable:
                relative_agent_pos[obs_w_multi_agent[obs], :] = furniture_env[obs].rel_ctl_pts_pos
                pass
            # relative_agent_pos[obs_w_multi_agent[obs], :] = furniture_env[0].global2relative(start_position[obs_w_multi_agent[obs]], obstacle_environment[obs])

        print(relative_agent_pos)
        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots(figsize=(10, 8))  # figsize=(10, 8)
        ax.set_aspect(1.0)
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

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

            weights = furniture_avoider.get_influence_weight_at_points(position_list[:, :, ii - 1], 3)
            print(f"weights: {weights}")

            for jj, furniture in enumerate(furniture_env):
                if furniture.is_controllable:
                    num_attractor = len(obs_w_multi_agent[jj])
                    global_attractor_pos = furniture.relative2global(relative_attractor_position, furniture.goal_container)
                    attractor_vel = np.zeros((num_attractor, dim))
                    for attractor in range(num_attractor):
                        attractor_vel[attractor, :] = attractor_dynamic.evaluate(global_attractor_pos[attractor, :])
                    attractor_weights = attractor_dynamic.get_weights_attractors(global_attractor_pos)
                    goal_vel, goal_rot = attractor_dynamic.get_goal_velocity(global_attractor_pos, attractor_vel,
                                                                             attractor_weights)
                    new_goal_pos = goal_vel * dt_step + furniture.goal_container.center_position
                    new_goal_ori = -goal_rot * dt_step + furniture.goal_container.orientation
                    furniture.goal_container.center_position = new_goal_pos
                    furniture.goal_container.orientation = new_goal_ori

                    global_attractor_pos = furniture.relative2global(relative_attractor_position, furniture.goal_container)
                    for i in obs_w_multi_agent[jj]:
                        furniture_avoider.set_attractor_position(global_attractor_pos[i], i)

            # loop
            for obs in range(num_obs):
                num_agents_in_obs = len(obs_w_multi_agent[obs])
                for agent in obs_w_multi_agent[obs]:
                    temp_env = furniture_avoider.env_slicer(obs)
                    velocity[agent, :] = furniture_avoider.evaluate_furniture(position_list[agent, :, ii - 1],
                                                                              agent, temp_env)
                    velocity[agent, :] = velocity[agent, :] * weights[obs][agent]

                obs_vel = np.zeros(2)
                if obs_w_multi_agent[obs]:
                    for agent in obs_w_multi_agent[obs]:
                        obs_vel += weights[obs][agent] * velocity[agent, :]
                else:
                    obs_vel = np.array([-0.3, 0.])

                angular_vel = np.zeros(num_agents_in_obs)
                for agent in obs_w_multi_agent[obs]:
                    angular_vel[agent] = weights[obs][agent] * np.cross(
                        (obstacle_environment[obs].center_position - position_list[agent, :, ii - 1]),
                        (velocity[agent, :] - obs_vel))

                angular_vel_obs = angular_vel.sum()
                obstacle_environment[obs].linear_velocity = obs_vel
                obstacle_environment[obs].angular_velocity = -angular_vel_obs
                obstacle_environment[obs].do_velocity_step(dt_step)
                for agent in obs_w_multi_agent[obs]:
                    position_list[agent, :, ii] = obstacle_environment[obs].transform_relative2global(
                        relative_agent_pos[agent, :])

            # Clear right before drawing again
            ax.clear()

            # Drawing and adjusting of the axis
            for agent in range(total_ctl_pts):
                plt.plot(position_list[agent, 0, :ii], position_list[agent, 1, :ii], ':',
                         color='#135e08')
                plt.plot(position_list[agent, 0, ii], position_list[agent, 1, ii],
                         'o', color='#135e08', markersize=12, )
                plt.arrow(position_list[agent, 0, ii], position_list[agent, 1, ii], velocity[agent, 0],
                          velocity[agent, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            for agent in range(total_ctl_pts):
                ax.plot(initial_dynamics[agent].attractor_position[0],
                        initial_dynamics[agent].attractor_position[1], 'k*', markersize=8, )
            ax.grid()

            ax.set_aspect('equal', adjustable='box')

            plt.pause(dt_sleep)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break


def multiple_robots():
    center_point = 1.
    num_agent = 2
    str_axis = "1.6,0.6".split(",")
    axis = [float(str_axis[0]), float(str_axis[1])]
    max_ax_len = max(axis)
    min_ax_len = min(axis)
    obstacle_pos = np.array([[center_point, 0.], [3.0, 0.]])

    rel_agent_pos, radius = calculate_relative_position(num_agent, max_ax_len, min_ax_len)

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(Cuboid(
        axes_length=[max_ax_len, min_ax_len],
        center_position=obstacle_pos[0],
        margin_absolut=0,
        orientation=math.pi/2,
        tail_effect=False,
        repulsion_coeff=1,
    ))
    obstacle_environment.append(Ellipse(
        axes_length=[0.6, 0.6],
        center_position=obstacle_pos[1],
        margin_absolut=radius,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([-0.3, 0.0]),
    ))

    agent_pos = relative2global(rel_agent_pos, obstacle_environment[0])

    attractor_env = ObstacleContainer()
    attractor_env.append(Cuboid(
        axes_length=[max_ax_len, min_ax_len],
        center_position=obstacle_pos[0],
        margin_absolut=0.,
        orientation=math.pi/2,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=np.array([0., 0.]),
    ))

    attractor_pos = relative2global(rel_agent_pos, attractor_env[0])

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
    for i in range(num_agent):
        initial_dynamics.append(
            LinearSystem(
                attractor_position=attractor_pos[i],
                maximum_velocity=1, distance_decrease=0.3
            )
        )

    obs_multi_agent = {0: [0, 1, 2], 1: []}
    obs_multi_agent = {0: [], 1: []}
    for i in range(num_agent):
        obs_multi_agent[0].append(i)

    furniture_env = []
    for i in range(1):
        furniture_env.append(
            Furniture(
                True,
                2,
                [max_ax_len, min_ax_len],
                "Cuboid",
                obstacle_pos[i],
                math.pi/2,
                np.array([0., 0.]),
                obstacle_pos[i],
                math.pi/2,
            )
        )

    furniture_env.append(
        Furniture(
            False,
            2,
            [max_ax_len, min_ax_len],
            "Ellipse",
            obstacle_pos[-1],
            0.,
            np.array([-0.3, 0.]),
            obstacle_pos[-1],
            0.,
        )
    )

    furniture_env[-1].furniture_container.margin_absolut = radius

    DynamicFurniture().run(
        furniture_env,
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        rel_agent_pos,
        attractor_env,
        x_lim=[-2, 4],
        y_lim=[-4, 4],
        dt_step=0.05,
    )


if __name__ == "__main__":
    plt.close('all')
    plt.ion()

    multiple_robots()
