from math import cos, sin
from dynamic_obstacle_avoidance.containers import ObstacleContainer
import matplotlib.pyplot as plt
from dynamic_obstacle_avoidance.obstacles import Ellipse
from vartools.dynamical_systems import DynamicalSystem
import numpy as np
from vartools.dynamical_systems import plot_dynamical_system_quiver, plot_dynamical_system_streamplot
from dynamic_obstacle_avoidance.visualization import plot_obstacles

pause = False


def onclick(event):
    global pause
    pause = not pause


def plot_dynamical_system(
        dynamical_system=None, x_lim=None, y_lim=None, n_resolution=15,
        figsize=(10, 7), plottype='quiver', axes_equal=True, fig_ax_handle=None,
        DynamicalSystem=None):
    """ Evaluate the dynamics of the dynamical system. """
    if DynamicalSystem is not None:
        raise Exception("'DynamicalSystem' -> Argument depreciated,"
                        + " use 'dynamical_system' instead.")
    dim = 2  # only for 2d implemented
    if x_lim is None:
        x_lim = [-10, 10]

    if y_lim is None:
        y_lim = [-10, 10]

    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], nx),
                                 np.linspace(y_lim[0], y_lim[1], ny))

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros(positions.shape)
    for it in range(positions.shape[1]):
        velocities[:, it] = dynamical_system.evaluate(positions[:, it])

    index_nonzero = np.unique(velocities.nonzero()[1])

    # plt.figure()
    if fig_ax_handle is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax_handle

    if plottype == 'quiver':
        ax.quiver(positions[0, index_nonzero], positions[1, index_nonzero],
                  velocities[0, index_nonzero], velocities[1, index_nonzero], color="blue")
    elif plottype == 'streamplot':
        ax.streamplot(
            x_vals, y_vals,
            velocities[0, :].reshape(nx, ny), velocities[1, :].reshape(nx, ny), color="blue")
    else:
        raise ValueError(f"Unknown plottype '{plottype}'.")

    if axes_equal:
        ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    plt.ion()
    plt.show()


class AttractorDynamics(DynamicalSystem):
    min_dist = 1
    dim = 2

    def __init__(self, agent, cutoff_dist=5, max_repulsion=1):
        self.agent = agent
        self.cutoff_dist = cutoff_dist
        self.max_repulsion = max_repulsion
        self.animation_paused = False

    def evaluate(self, position):
        dir_agent = position - self.agent.center_position
        print(f"position before gamma: {position}")
        dist_agent = self.agent.get_gamma(position, in_global_frame=True, )

        if dist_agent < 1:
            # raise Warning("attractor got run over :'(")
            print("attractor got run over :'(")
            # TODO: no theoretical value
            return np.zeros(self.dim)

        unit_dir_agent = dir_agent / np.linalg.norm(dir_agent)
        unit_lin_vel = self.agent.linear_velocity / np.linalg.norm(self.agent.linear_velocity)
        max_repulsion = np.dot(unit_dir_agent, unit_lin_vel)
        if max_repulsion <= 0.:
            return np.zeros(self.dim)

        slope = (-max_repulsion) / (self.cutoff_dist - self.min_dist)
        offset = max_repulsion - (slope * self.min_dist)
        repulsion_magnitude = (slope * dist_agent) + offset
        if repulsion_magnitude <= 0.:
            return np.zeros(self.dim)
        vect_agent = repulsion_magnitude * unit_dir_agent

        perpendicular_vect_agent = np.array([-unit_lin_vel[1], unit_lin_vel[0]])
        basis_e = np.array([unit_lin_vel, perpendicular_vect_agent])
        lambda_p = 10
        basis_d = np.eye(self.dim)
        basis_d[1] *= lambda_p
        new_vect = basis_e @ basis_d @ basis_e.T @ vect_agent

        return new_vect


def main():
    obs_env = ObstacleContainer()
    pos = np.array([0., 0.])
    vel = np.array([1., 0.])
    # u_vel = vel / np.linalg.norm(vel)
    obstacle_environment = Ellipse(
        axes_length=[0.6, 0.6],
        center_position=pos,
        margin_absolut=0.2,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1,
        linear_velocity=vel,
    )
    obs_env.append(obstacle_environment)
    my_dynamics = AttractorDynamics(obstacle_environment)
    x_lim = [-5, 5]
    y_lim = x_lim
    fig, ax = plt.subplots(figsize=(10, 8))

    theta = np.deg2rad(10)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    for ii in range(37):
        fig.canvas.mpl_connect('button_press_event', onclick)
        if pause:
            plt.pause(7)
            continue

        ax.clear()
        plot_dynamical_system(n_resolution=61, dynamical_system=my_dynamics, x_lim=x_lim, y_lim=y_lim,
                              fig_ax_handle=[fig, ax])
        plot_obstacles(ax, obs_env, x_lim, y_lim, showLabel=False, drawVelArrow=False)
        plt.arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
        vel = np.dot(rot, vel)
        obstacle_environment.linear_velocity = vel
        plt.pause(0.5)


if __name__ == "__main__":
    plt.close("all")
    main()
