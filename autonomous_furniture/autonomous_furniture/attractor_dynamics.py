from math import cos, sin
import warnings
from dynamic_obstacle_avoidance.containers import ObstacleContainer
import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles import Ellipse, Cuboid, GammaType
from vartools.dynamical_systems import DynamicalSystem
import numpy as np
from numpy import linalg as LA
from vartools.dynamical_systems import (
    plot_dynamical_system_quiver,
    plot_dynamical_system_streamplot,
)
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.avoidance.modulation import (
    obs_avoidance_interpolation_moving,
)

pause = True


def onclick(event):
    global pause
    pause = not pause


def get_weight_from_gamma(
    gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5
):
    weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
    weights = weights / frac_gamma_nth
    weights = 1.0 / weights
    weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
    weights = weights / n_points
    return weights


def plot_dynamical_system(
    dynamical_system=None,
    x_lim=None,
    y_lim=None,
    n_resolution=15,
    figsize=(10, 7),
    plottype="quiver",
    axes_equal=True,
    fig_ax_handle=None,
    DynamicalSystem=None,
    label=None,
):
    """Evaluate the dynamics of the dynamical system."""
    if DynamicalSystem is not None:
        raise Exception(
            "'DynamicalSystem' -> Argument depreciated,"
            + " use 'dynamical_system' instead."
        )
    dim = 2  # only for 2d implemented
    if x_lim is None:
        x_lim = [-10, 10]

    if y_lim is None:
        y_lim = [-10, 10]

    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros(positions.shape)
    for it in range(positions.shape[1]):
        velocities[:, it] = dynamical_system.evaluate(positions[:, it], None)

    index_nonzero = np.unique(velocities.nonzero()[1])

    # plt.figure()
    if fig_ax_handle is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax_handle

    if plottype == "quiver":
        ax.quiver(
            positions[0, index_nonzero],
            positions[1, index_nonzero],
            velocities[0, index_nonzero],
            velocities[1, index_nonzero],
            color="#9673A6",
            label=label,
        )
    elif plottype == "streamplot":
        ax.streamplot(
            x_vals,
            y_vals,
            velocities[0, :].reshape(nx, ny),
            velocities[1, :].reshape(nx, ny),
            color="blue",
        )
    else:
        raise ValueError(f"Unknown plottype '{plottype}'.")

    if axes_equal:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    plt.ion()
    plt.show()


class AttractorDynamics(DynamicalSystem):
    min_dist = 1
    dim = 2

    def __init__(
        self, env, cutoff_dist=5, max_repulsion=1, parking_zone=None, num_goals=None
    ):
        self.env = env
        self.cutoff_dist = cutoff_dist
        self.max_repulsion = max_repulsion
        self.parking_zone = parking_zone
        if self.parking_zone is None:
            self.move_to_pk = False
        else:
            self.move_to_pk = True
        self.go_to_pk = [False] * len(self.parking_zone)
        # self.go_to_pk = False
        self.state = ["idle"] * len(self.parking_zone)
        self.animation_paused = False
        self.lambda_p = 10

    def evaluate(self, position, furniture_eval):
        dir_agent = position - self.env[-1].center_position
        dist_agent = self.env[-1].get_gamma(
            position,
            in_global_frame=True,
        )

        if dist_agent < 1:
            # raise Warning("attractor got run over :'(")
            print("attractor got run over :'(")
            # TODO: no theoretical value
            return np.zeros(self.dim)

        if furniture_eval is not None:
            temp_env = self.env[0:furniture_eval] + self.env[furniture_eval + 1 : -1]
        else:
            temp_env = self.env[:-1]

        unit_dir_agent = dir_agent / np.linalg.norm(dir_agent)
        unit_lin_vel = self.env[-1].linear_velocity / np.linalg.norm(
            self.env[-1].linear_velocity
        )
        max_repulsion = np.dot(unit_dir_agent, unit_lin_vel)

        if self.state[furniture_eval] == "idle":
            if max_repulsion <= 0.0:
                self.state[furniture_eval] = "idle"
                return np.zeros(self.dim), self.go_to_pk

            slope = (-max_repulsion) / (self.cutoff_dist - self.min_dist)
            offset = max_repulsion - (slope * self.min_dist)
            repulsion_magnitude = (slope * dist_agent) + offset

            if repulsion_magnitude <= 0.0:
                self.state[furniture_eval] = "idle"
                return np.zeros(self.dim), self.go_to_pk
            else:
                self.state[furniture_eval] = "avoiding"

            vect_agent = repulsion_magnitude * unit_dir_agent
            perpendicular_vect_agent = np.array([-unit_lin_vel[1], unit_lin_vel[0]])
            basis_e = np.array([unit_lin_vel, perpendicular_vect_agent])
            basis_d = np.eye(self.dim)
            basis_d[1] *= self.lambda_p
            new_vect = basis_e.T @ basis_d @ basis_e @ vect_agent
            mod_vel = self.avoid(position, new_vect, temp_env)
            self.go_to_pk[furniture_eval] = False
            # mod_vel = np.zeros(self.dim)

        elif self.state[furniture_eval] == "avoiding":
            # unit_dir_agent = dir_agent / np.linalg.norm(dir_agent)
            # unit_lin_vel = self.env[-1].linear_velocity / np.linalg.norm(self.env[-1].linear_velocity)
            # max_repulsion = np.dot(unit_dir_agent, unit_lin_vel)

            if max_repulsion <= 0.0:
                self.state[furniture_eval] = "parking"
                return np.zeros(self.dim), self.go_to_pk

            slope = (-max_repulsion) / (self.cutoff_dist - self.min_dist)
            offset = max_repulsion - (slope * self.min_dist)
            repulsion_magnitude = (slope * dist_agent) + offset

            if max_repulsion <= 0.0:
                self.state[furniture_eval] = "parking"
                return np.zeros(self.dim), self.go_to_pk

            vect_agent = repulsion_magnitude * unit_dir_agent
            perpendicular_vect_agent = np.array([-unit_lin_vel[1], unit_lin_vel[0]])
            basis_e = np.array([unit_lin_vel, perpendicular_vect_agent])
            basis_d = np.eye(self.dim)
            basis_d[1] *= self.lambda_p
            new_vect = basis_e.T @ basis_d @ basis_e @ vect_agent
            mod_vel = self.avoid(position, new_vect, temp_env)
            self.go_to_pk[furniture_eval] = False

        elif self.state[furniture_eval] == "parking":
            mod_vel = np.zeros(self.dim)
            self.go_to_pk[furniture_eval] = True

        else:
            mod_vel = np.zeros(self.dim)
            self.go_to_pk[furniture_eval] = False

        # if self.go_to_pk is False:
        #     unit_dir_agent = dir_agent / np.linalg.norm(dir_agent)
        #     unit_lin_vel = self.env[-1].linear_velocity / np.linalg.norm(self.env[-1].linear_velocity)
        #     max_repulsion = np.dot(unit_dir_agent, unit_lin_vel)
        #     if max_repulsion <= 0.:
        #         # self.state[furniture_eval] = "idle"
        #         return np.zeros(self.dim)
        #
        #     slope = (-max_repulsion) / (self.cutoff_dist - self.min_dist)
        #     offset = max_repulsion - (slope * self.min_dist)
        #     repulsion_magnitude = (slope * dist_agent) + offset
        #     if repulsion_magnitude <= 0.:
        #         # self.state[furniture_eval] = "idle"
        #         return np.zeros(self.dim)
        #     vect_agent = repulsion_magnitude * unit_dir_agent
        #     # self.state[furniture_eval] = "avoiding"
        #     perpendicular_vect_agent = np.array([-unit_lin_vel[1], unit_lin_vel[0]])
        #     basis_e = np.array([unit_lin_vel, perpendicular_vect_agent])
        #     lambda_p = 10
        #     basis_d = np.eye(self.dim)
        #     basis_d[1] *= self.lambda_p
        #     new_vect = basis_e.T @ basis_d @ basis_e @ vect_agent
        #     mod_vel = self.avoid(position, new_vect, temp_env)
        #
        # elif self.go_to_pk is True:
        #     gamma_list = np.zeros(len(self.parking_zone))
        #     for ii, pk in enumerate(self.parking_zone):
        #         gamma_list[ii] = self.env[furniture_eval].get_gamma(pk.center_position, in_global_frame=True,)
        #
        #     parking_zone_index = np.argmin(gamma_list)
        #     mod_vel = self.parking_zone[parking_zone_index].center_position - self.env[furniture_eval].center_position

        # else:   # WTF ?
        #     gamma_list = np.zeros(len(self.parking_zone))
        #     for ii, pk in enumerate(self.parking_zone):
        #         gamma_list[ii] = self.env[furniture_eval].get_gamma(pk.center_position, in_global_frame=True,)
        #
        #     parking_zone_index = np.argmin(gamma_list)
        #     dir_parking_zone = self.parking_zone[parking_zone_index].center_position - position
        #     norm_parking_zone = np.linalg.norm(dir_parking_zone)
        #     if norm_parking_zone > 1:
        #         new_vect = dir_parking_zone / norm_parking_zone
        #     elif norm_parking_zone < 0.3:
        #         new_vect = np.zeros(self.dim)
        #     else:
        #         new_vect = dir_parking_zone
        #
        #     mod_vel = self.avoid(position, new_vect, temp_env)

        # return mod_vel
        return mod_vel, self.go_to_pk

    def print_state(self, index):
        print(f"state of fur {index}: {self.state[index]}")

    def avoid(
        self,
        position: np.ndarray,
        initial_velocity: np.ndarray,
        env,
        const_speed: bool = True,
    ) -> np.ndarray:
        vel = obs_avoidance_interpolation_moving(
            position=position, initial_velocity=initial_velocity, obs=env
        )

        if const_speed:
            vel_mag = LA.norm(vel)
            if vel_mag:
                vel = vel / vel_mag * LA.norm(initial_velocity)

        # elif self.maximum_speed is not None:
        #     vel_mag = LA.norm(vel)
        #     if vel_mag > self.maximum_speed:
        #         vel = vel / vel_mag * self.maximum_speed

        return vel

    def set_lambda(self, lambda_p):
        self.lambda_p = lambda_p

    def get_gamma_product_attractor(
        self, position, env, gamma_type=GammaType.EUCLEDIAN
    ):
        if not len(env):
            # Very large number
            return 1e20
        gamma_list = np.zeros(len(env))
        for ii, obs in enumerate(env):
            # gamma_type needs to be implemented for all obstacles
            gamma_list[ii] = obs.get_gamma(
                position,
                in_global_frame=True
                # , gamma_type=gamma_type
            )

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            warnings.warn("Collision detected.")
            # breakpoint()
            return 0

        gamma = np.min(gamma_list)

        if np.isnan(gamma):
            breakpoint()
        return gamma

    def get_weights_attractors(
        self, attractor_points, furniture_eval, gamma_type=GammaType.EUCLEDIAN
    ):
        num_attractors = attractor_points.shape[0]

        temp_env = self.env[0:furniture_eval] + self.env[furniture_eval + 1 :]

        gamma_list = np.zeros(num_attractors)
        # for the moment only take into account the person/qolo/agent
        for ii in range(num_attractors):
            # gamma_list[ii] = self.agent.get_gamma(
            #     attractor_points[ii, :], in_global_frame=True, gamma_type=gamma_type
            # )
            gamma_list[ii] = self.get_gamma_product_attractor(
                attractor_points[ii, :], temp_env
            )

        attractor_weights = np.zeros(gamma_list.shape)
        ind_nonzero = gamma_list < self.cutoff_dist
        if not any(ind_nonzero):
            attractor_weights = np.full(num_attractors, 1 / num_attractors)
            return attractor_weights

        attractor_weights[ind_nonzero] = get_weight_from_gamma(
            gamma_list[ind_nonzero],
            cutoff_gamma=self.cutoff_dist,
            n_points=num_attractors,
        )

        attractor_weights_sum = np.sum(attractor_weights)
        if attractor_weights_sum > 1:
            attractor_weights = attractor_weights / attractor_weights_sum
        else:
            attractor_weights[-1] = 1 - attractor_weights_sum

        return attractor_weights

    def get_goal_velocity(
        self, attractor_pos, attractor_velocities, attractor_weights, furniture_eval
    ):
        num_attractors = attractor_weights.shape[0]
        lin_vel = np.zeros(2)
        for ii in range(num_attractors):
            lin_vel += attractor_weights[ii] * attractor_velocities[ii, :]

        angular_vel = np.zeros(num_attractors)
        for attractor in range(num_attractors):
            angular_vel[attractor] = attractor_weights[attractor] * np.cross(
                (
                    self.env[furniture_eval].center_position
                    - attractor_pos[attractor, :]
                ),
                (attractor_velocities[attractor, :] - lin_vel),
            )
        angular_vel_sum = angular_vel.sum()

        return lin_vel, angular_vel_sum

    def get_gamma_at_attractor(self, attractor, obstacle):
        gamma_values = np.zeros(attractor.shape[0])
        for ii in range(attractor.shape[0]):
            gamma_values[ii] = self.get_gamma_product_attractor(
                attractor[ii, :], obstacle
            )
        return gamma_values


def main():
    obs_env = ObstacleContainer()
    pos = np.array([[0.0, 0.0], [2.0, 2.0]])
    vel = np.array([1.0, 0.0])
    # u_vel = vel / np.linalg.norm(vel)
    theta = np.deg2rad(10)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    # vel = np.dot(rot, vel)

    obstacle_environment = Ellipse(
        axes_length=[0.6, 0.6],
        center_position=pos[0],
        margin_absolut=0.0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=vel,
    )
    obs_env.append(
        Cuboid(
            axes_length=[1.6, 0.6],
            center_position=pos[1],
            margin_absolut=0.0,
            orientation=0,
            tail_effect=False,
            repulsion_coeff=1,
            linear_velocity=vel,
        )
    )
    obs_env.append(obstacle_environment)

    parking_zone_cp = np.array([[1.0, 1.0]])
    parking_zone = ObstacleContainer()
    for pk in range(len(parking_zone_cp)):
        parking_zone.append(
            Cuboid(
                axes_length=obs_env[pk].axes_length,
                center_position=parking_zone_cp[pk],
                margin_absolut=0,
                orientation=0.0,
                tail_effect=False,
                repulsion_coeff=1,
                linear_velocity=np.array([0.0, 0.0]),
            )
        )

    my_dynamics = AttractorDynamics(obs_env, parking_zone=parking_zone)
    x_lim = [-5, 5]
    y_lim = x_lim
    fig, ax = plt.subplots()

    label_qui = "$v^a$"
    label_arr = "$v^p$"

    ii = 0
    while ii < 9:
        fig.canvas.mpl_connect("button_press_event", onclick)
        if pause:
            plt.pause(0.1)
            continue

        ax.clear()
        # my_dynamics.set_lambda(ii+1)
        plot_dynamical_system(
            n_resolution=16,
            dynamical_system=my_dynamics,
            x_lim=x_lim,
            y_lim=y_lim,
            fig_ax_handle=[fig, ax],
            label=label_qui,
        )
        plot_obstacles(ax, obs_env, x_lim, y_lim, showLabel=False, drawVelArrow=False)
        plt.arrow(
            pos[0, 0],
            pos[0, 1],
            vel[0],
            vel[1],
            linewidth=3,
            head_width=0.1,
            head_length=0.1,
            fc="#6C8EBF",
            ec="#6C8EBF",
            color="#6C8EBF",
            label=label_arr,
        )
        plt.legend()
        vel = np.dot(rot, vel)
        obstacle_environment.linear_velocity = vel
        obs_env[-1].linear_velocity = vel
        plt.pause(0.5)
        ii += 1

        if not plt.fignum_exists(fig.number):
            print("Stopped animation on closing of the figure..")
            break


if __name__ == "__main__":
    plt.close("all")
    main()
