import time
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from vartools.states import Pose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.obstacles import get_intersection_position
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.visualization.plot_qolo import integrate_with_qolo
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.dynamics.segmented_dynamics import create_segment_from_points
from nonlinear_avoidance.dynamics.segmented_dynamics import WavyPathFollowing

from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)
from nonlinear_avoidance.nonlinear_rotation_avoider import (
    SingularityConvergenceDynamics,
)

# from autonomous_furniture.rviz_animator import RvizSimulator
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

# Local import
# from models import SimpleAgent,
from models import RvizTable, RvizQolo
from models import update_shapes_of_agent
from models import AgentTransformBroadCaster
from models import AgentContainer


class GrassPublisher(Node):
    line_length = 1.0
    x_lim = [-5, 5]
    position = [[0.0, -8.0], [0.0, 8.0], [-6.0, 4.0], [6.0, -4.0]]
    axes_length = [[20.0, 4.0], [20.0, 4.0], [8.0, 12.0], [8.0, 12.0]]

    def __init__(self, x_lim=[-5, 5], y_lim=[-5, 5]):
        super().__init__("environment_publisher")
        self.markers_array = MarkerArray()

        self.publisher_ = self.create_publisher(MarkerArray, "environment", 3)
        self.markers_array.markers.append(self.create_ground())

        for ii, (pos, axes) in enumerate(zip(self.position, self.axes_length)):
            self.markers_array.markers.append(
                self.create_grass(
                    x_pos=pos[0],
                    y_pos=pos[1],
                    delta_x=axes[0],
                    delta_y=axes[1],
                    ns=f"grass{ii}",
                )
            )
        # self.markers_array.markers.append(
        #     self.create_grass(x_pos=0.0, y_pos=-8.0, delta_x=20, delta_y=4, ns="grass0")
        # )
        # self.markers_array.markers.append(
        #     self.create_grass(x_pos=0.0, y_pos=8.0, delta_x=20, delta_y=4, ns="grass1")
        # )
        # self.markers_array.markers.append(
        #     self.create_grass(x_pos=-6, y_pos=4, delta_x=8, delta_y=12, ns="grass2")
        # )
        # self.markers_array.markers.append(
        #     self.create_grass(x_pos=6, y_pos=-4, delta_x=8, delta_y=12, ns="grass3")
        # )

        # self.place_lines(n_lines=5)
        # Line
        self.publisher_.publish(self.markers_array)

    def place_lines(self, n_lines: int) -> None:
        x_range = self.x_lim[1] - self.x_lim[0]
        delta_x = (x_range - self.line_length) / (n_lines - 1)

        for ii in range(5):
            name = f"line{ii}"
            x_pos = delta_x * ii + self.line_length * 0.5 + self.x_lim[0]
            print(x_pos)
            self.markers_array.markers.append(
                self.create_center_line(x_pos=x_pos, y_pos=0.0, ns=name)
            )

    def create_ground(self, ns="ground"):
        marker = Marker()
        # def publish_cube(self):
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 0
        marker.type = 1  # is cube
        # marker.action = visualization_msgs::Marker::ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 20.0
        marker.scale.y = 20.0
        marker.scale.z = 0.01
        # 255, 87, 51.
        marker.color.a = 1.0
        marker.color.r = 211 / 256.0
        marker.color.g = 211 / 256.0
        marker.color.b = 211 / 256.0
        return marker

    def create_grass(
        self, x_pos=0.0, y_pos=-4.0, delta_x=10.0, delta_y=3.0, ns="grass"
    ):
        # def publish_cube(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 0
        marker.type = 1  # is cube
        # marker.action = visualization_msgs::Marker::ADD
        marker.pose.position.x = float(x_pos)
        marker.pose.position.y = float(y_pos)
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(delta_x)
        marker.scale.y = float(delta_y)
        marker.scale.z = 0.2
        # 255, 87, 51.
        marker.color.a = 1.0
        marker.color.r = 100 / 256.0
        marker.color.g = 252 / 256.0
        marker.color.b = 20 / 256.0
        return marker

    def create_center_line(self, x_pos=0.0, y_pos=0.0, ns="line"):
        # def publish_cube(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 0
        marker.type = 1  # is cube
        # marker.action = visualization_msgs::Marker::ADD
        marker.pose.position.x = x_pos
        marker.pose.position.y = y_pos
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 0.2
        marker.scale.z = 0.02
        # 255, 87, 51.
        marker.color.a = 1.0
        marker.color.r = 255 / 256.0
        marker.color.g = 255 / 256.0
        marker.color.b = 255 / 256.0
        return marker


class RvizQoloAnimator(Animator):
    initial_dynamics = create_segment_from_points(
        [[-7.0, -4.0], [0.0, -4.0], [0.0, 4.0], [7.5, 4.0]]
    )

    def setup(
        self,
        broadcaster: Optional[AgentTransformBroadCaster] = None,
        do_plotting: bool = True,
        x_lim=[-5, 5],
        y_lim=[-5, 5],
        container=None,
    ) -> None:
        self.x_lim = x_lim
        self.y_lim = y_lim

        start_pose = Pose(
            position=self.initial_dynamics.segments[0].start, orientation=0.0
        )

        self.robot = RvizQolo(pose=start_pose)
        self.robot.required_margin = 0.7

        if container is None:
            intersecting_id, reference_point = self.create_agent_container()
        else:
            self.agent_container = container
            intersecting_id = []
        # breakpoint()

        rotation_projector = ProjectedRotationDynamics(
            attractor_position=self.initial_dynamics.segments[-1].end,
            initial_dynamics=self.initial_dynamics,
            # reference_velocity=lambda x: x - center_velocity.center_position,
        )

        self.avoider = SingularityConvergenceDynamics(
            initial_dynamics=self.initial_dynamics,
            # convergence_system=convergence_dynamics,
            obstacle_environment=self.agent_container.get_obstacles(
                desired_margin=self.robot.required_margin
            ),
            obstacle_convergence=rotation_projector,
        )

        self.broadcaster = broadcaster

        self.do_plotting = do_plotting
        if do_plotting:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = None

        # Initial update of transform
        update_shapes_of_agent(self.robot)
        for ii, agent in enumerate(self.agent_container):
            update_shapes_of_agent(agent)

            if ii in intersecting_id:
                agent.shapes[0].margin_absolut = self.robot.required_margin
                agent.shapes[0].set_reference_point(
                    reference_point, in_global_frame=True
                )

    def create_agent_container(self):
        self.agent_container = AgentContainer()
        self.agent_container.append(
            RvizTable(pose=Pose(position=[-3.6, -3.7], orientation=0.4 * np.pi))
        )
        self.agent_container.append(
            RvizTable(pose=Pose(position=[4.0, 3.3], orientation=0.4 * np.pi))
        )

        reference_point = np.array([0.4, 0.7])
        intersecting_id = [self.agent_container.n_obstacles]
        self.agent_container.append(
            RvizTable(pose=Pose(position=[1.5, 0.4], orientation=-0.1 * np.pi))
        )
        intersecting_id.append(self.agent_container.n_obstacles)
        self.agent_container.append(
            RvizTable(pose=Pose(position=[0.3, 1.3], orientation=0.4 * np.pi))
        )

        self.agent_container.append(
            RvizTable(pose=Pose(position=[-0.6, -2.6], orientation=-0.1 * np.pi))
        )
        # self.agent_container.append(
        #     RvizTable(pose=Pose(position=[0.7, 4.8], orientation=-0.3 * np.pi))
        # )

        return intersecting_id, reference_point

    def update_step(self, ii):
        print("ii", ii)
        # self.robot.twist.linear = self.avoider.avoid(
        #     position=self.robot.pose.position,
        #     obstacle_list=self.agent_container.get_obstacles(
        #         desired_margin=self.robot.required_margin
        #     ),
        # )
        self.robot.twist.linear = self.avoider.evaluate_sequence(
            self.robot.pose.position
        )

        self.robot.update_step(dt=self.dt_simulation)
        update_shapes_of_agent(self.robot)
        if self.broadcaster is not None:
            self.broadcaster.broadcast([self.robot])
            self.broadcaster.broadcast(self.agent_container)

        if not self.do_plotting:
            return

        self.ax.clear()
        plot_obstacles(ax=self.ax, obstacle_container=self.robot.shapes, noTicks=False)
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.agent_container.get_obstacles(
                desired_margin=self.robot.required_margin
            ),
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            noTicks=False,
        )

        for segment in self.initial_dynamics.segments:
            self.ax.plot(
                [segment.start[0], segment.end[0]],
                [segment.start[1], segment.end[1]],
                marker="o",
            )


class RosAnimatorNode(Node):
    @property
    def period(self) -> float:
        return 1.0 / self.animator.dt_simulation

    def __init__(self, animator):
        self.animator = animator
        self.it = 0
        self.timer = self.create_timer(self.period, self.update_step)

    def update_step(self):
        self.animator.update_step(self.it)
        self.it += 1


def main_wavy(
    it_max: int = 1000,
    delta_time: float = 0.1,
    do_ros: bool = True,
    # do_plotting: bool = False,
    do_plotting: bool = True,
):
    if do_ros:
        broadcaster = AgentTransformBroadCaster()
        publisher = GrassPublisher()
    else:
        broadcaster = None

    animator = RvizQoloAnimator(
        it_max=it_max,
        dt_simulation=delta_time,
        dt_sleep=delta_time,
    )
    animator.setup(
        broadcaster=broadcaster,
        do_plotting=do_plotting,
        x_lim=[-10, 10],
        y_lim=[-10, 10],
    )

    # Create launch rviz
    nodes = []
    for agent in animator.agent_container:
        nodes.append(agent.get_node())

    print("Done Launching")
    if do_plotting:
        animator.run()
    else:
        for ii in range(it_max):
            animator.update_step(ii)
            time.sleep(delta_time)

    print("End of script")


def plot_grass(ax, publisher):
    for ii, (pos, axes) in enumerate(zip(publisher.position, publisher.axes_length)):
        pos_edge = np.array(pos) - np.array(axes) * 0.5
        grass = patches.Rectangle(pos_edge, axes[0], axes[1], color="black", zorder=-3)
        grass.set(color="#D3D3D3")
        ax.add_artist(grass)


def create_container_from_grass(publisher):
    container = RotationContainer()
    for ii, (pos, axes) in enumerate(zip(publisher.position, publisher.axes_length)):
        container.append(Cuboid(pose=Pose(pos), axes_length=axes))
    return container


def create_new_table(
    environment,
    x_lim,
    y_lim,
    start,
    goal,
    it_max=100,
    margin_absolut=0.7,
):
    grass_container = create_container_from_grass(publisher=GrassPublisher)

    for ii in range(100):
        pose = np.random.rand(3)
        pose[0] = pose[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
        pose[1] = pose[1] * (y_lim[1] - x_lim[0]) + x_lim[0]
        pose[2] = pose[2] * 2 * np.pi - np.pi

        gamma = grass_container.get_minimum_gamma(pose[:2])
        if gamma <= 1:
            continue

        table = RvizTable(pose=Pose(position=pose[:2], orientation=pose[2]))
        update_shapes_of_agent(table)

        colliding = False
        for obs in table.shapes:
            obs.margin_absolut = margin_absolut

            if obs.get_gamma(start, in_global_frame=True) <= 1:
                colliding = True
                break

            if obs.get_gamma(goal, in_global_frame=True) <= 1:
                colliding = True
                break

        if colliding:
            continue

        return table

    warnings.warn("Not found after too many iterations --- quitting...")
    return None


def update_references(obstacle, environment) -> bool:
    """Returns False if there is a 'double' intersection."""
    reference_position = None
    obs_other = None
    for other in environment:
        intersection = get_intersection_position(obstacle, other)

        if intersection is None:
            continue

        if reference_position is not None:
            warnings.warn("Reference point already replaced ignoring..")
            return False

        reference_position = intersection
        obs_other = other

        if not np.allclose(
            other.get_reference_point(in_global_frame=True), other.center_position
        ):
            warnings.warn("Reference point already replaced ignoring..")
            return False

    if reference_position is None:
        return True

    obs_other.set_reference_point(reference_position, in_global_frame=True)
    obstacle.set_reference_point(reference_position, in_global_frame=True)
    return True


def random_placement_tables(x_lim, y_lim, start, goal, n_tables=7) -> AgentContainer:
    container = AgentContainer()
    for tt in range(n_tables):
        environment = container.get_obstacles()
        new_table = create_new_table(
            environment, x_lim=x_lim, y_lim=y_lim, start=start, goal=goal
        )

        successfull_reference = update_references(new_table.shapes[0], environment)

        if not successfull_reference:
            continue

        container.append(new_table)

    print(f"Placed tables: {container.n_obstacles} / {n_tables}")
    return container


class SwitchingDynamics:
    def __init__(self, segments, width: float = 1.0):

        self._mid_points = [ss.end for ss in segments]
        self._local_attractors = [
            np.array([2, -3.5]),
            np.array([0, 6.0]),
            segments[-1].end,
        ]
        self.attractor_position = np.array(self._local_attractors[-1])
        # self.x_limit_1 = -2
        # self.y_limit_2 = 2

        self.switch_direction = np.array([1, 1])

        self.slowdown_distance = 1.0
        self.maximum_speed = 1.0

    def get_direction(self, position: np.ndarray) -> np.ndarray:
        if np.dot(self.switch_direction, (self._mid_points[1] - position)) < 0:
            return LinearSystem(self._local_attractors[2]).evaluate(position)

        if np.dot(self.switch_direction, (self._mid_points[0] - position)) < 0:
            return LinearSystem(self._local_attractors[1]).evaluate(position)

        return LinearSystem(self._local_attractors[0]).evaluate(position)

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        direction = self.get_direction(position)

        dist = np.linalg.norm(self.attractor_position - position)
        if dist > self.slowdown_distance:
            vel_max = self.maximum_speed
        else:
            vel_max = self.maximum_speed * dist / self.slowdown_distance

        if vel_max == 0:
            return np.zeros_like(position)

        return direction / np.linalg.norm(direction) * vel_max


class SwitchingDynamicsPathFollowing:
    def __init__(self, segments, width: float = 1.0):

        self.segments = segments
        self._mid_points = [ss.end for ss in segments]
        self._local_attractors = [
            np.array([2, -3.5]),
            np.array([0, 6.0]),
            segments[-1].end,
        ]
        self.attractor_position = np.array(self._local_attractors[-1])

        self.switch_direction = np.array([1, 1])

        self.slowdown_distance = 1.0
        self.maximum_speed = 1.0

    def get_direction(self, position: np.ndarray) -> np.ndarray:
        if np.dot(self.switch_direction, (self._mid_points[1] - position)) < 0:
            return WavyPathFollowing([self.segments[2]]).evaluate(position)

        if np.dot(self.switch_direction, (self._mid_points[0] - position)) < 0:
            return WavyPathFollowing([self.segments[1]]).evaluate(position)

        return WavyPathFollowing([self.segments[0]]).evaluate(position)

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        direction = self.get_direction(position)

        dist = np.linalg.norm(self.attractor_position - position)
        if dist > self.slowdown_distance:
            vel_max = self.maximum_speed
        else:
            vel_max = self.maximum_speed * dist / self.slowdown_distance

        if vel_max == 0:
            return np.zeros_like(position)

        return direction / np.linalg.norm(direction) * vel_max


def plot_vectorfield_nonlinear_global(n_grid=20, save_figure=False):
    from nonlinear_avoidance.visualization.plot_qolo import integrate_with_qolo

    # x_lim = [-10, 10.0]
    # y_lim = [-10.0, 10.]

    x_lim = [-6.5, 8.5]
    y_lim = [-7.0, 7.0]

    # x_lim = [-2.0, -0.5]
    # y_lim = [-4.5, -3.0]
    # figsize = (4, 3.5)
    figsize = (8, 7.0)
    figtype = ".pdf"
    position_start = np.array([-5.5, -4.0])

    np.random.seed(10)
    container = random_placement_tables(
        x_lim=x_lim,
        y_lim=y_lim,
        start=position_start,
        goal=RvizQoloAnimator.initial_dynamics.attractor_position,
    )
    # container = None

    animator = RvizQoloAnimator()
    animator.setup(x_lim=[-10, 10], y_lim=[-10, 10], container=container)
    # avoider = animator.avoider

    it_max = 1000

    plt.close("all")

    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        obstacle_container=animator.agent_container.get_obstacles(),
        dynamics=animator.avoider.evaluate_sequence,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        # attractor_position=dynamic.attractor_position,
        # do_quiver=False,
        do_quiver=True,
        show_ticks=False,
    )
    plot_obstacles(
        ax=ax,
        obstacle_container=animator.agent_container.get_obstacles(),
        x_lim=x_lim,
        y_lim=y_lim,
        # draw_reference=True,
    )
    plot_grass(ax=ax, publisher=GrassPublisher)

    trajectory = integrate_with_qolo(
        position_start,
        animator.avoider.evaluate_sequence,
        it_max=it_max,
        dt=0.03,
        ax=ax,
        attractor_position=animator.initial_dynamics.attractor_position,
    )

    converged = np.allclose(
        trajectory[:, -1], animator.initial_dynamics.attractor_position, atol=1e-1
    )
    print()
    print(f"Has converged: {converged}")
    print()
    if True:
        warnings.warn("No saving of figure...")
        return

    if save_figure:
        fig_name = "qolo_along_wavy_road_avoiding"
        fig.savefig("media/" + fig_name + figtype, bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        obstacle_container=[],
        dynamics=animator.initial_dynamics.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        # attractor_position=dynamic.attractor_position,
        do_quiver=False,
        show_ticks=False,
    )
    integrate_with_qolo(
        position_start,
        animator.initial_dynamics.evaluate,
        it_max=it_max,
        dt=0.03,
        ax=ax,
        attractor_position=dynamics.attractor_position,
    )
    plot_grass(ax=ax, publisher=GrassPublisher)

    if save_figure:
        fig_name = "qolo_along_wavy_road_initial"
        fig.savefig("media/" + fig_name + figtype, bbox_inches="tight", dpi=300)

    breakpoint()


def plot_switching_linear_dynamics(n_grid=10, save_figure=False):
    x_lim = [-6.5, 8.5]
    y_lim = [-7.0, 7.0]

    # figsize = (4, 3.5)
    figsize = (8, 7.0)
    figtype = ".pdf"
    position_start = np.array([-5, -4.0])
    it_max = 1000

    dynamics = SwitchingDynamics(RvizQoloAnimator.initial_dynamics.segments)

    # container = None
    container = random_placement_tables(
        x_lim=x_lim, y_lim=y_lim, start=position_start, goal=dynamics.attractor_position
    )

    animator = RvizQoloAnimator()
    animator.setup(x_lim=[-10, 10], y_lim=[-10, 10], container=container)
    plt.close("all")

    animator.avoider = ModulationAvoider(
        initial_dynamics=dynamics,
        obstacle_environment=animator.agent_container.get_obstacles(),
    )

    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        obstacle_container=[],
        # dynamics=dynamics.evaluate,
        dynamics=animator.avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        # attractor_position=dynamic.attractor_position,
        do_quiver=True,
        show_ticks=True,
    )
    plot_obstacles(
        ax=ax,
        obstacle_container=animator.agent_container.get_obstacles(),
        x_lim=x_lim,
        y_lim=y_lim,
        draw_reference=True,
    )
    plot_grass(ax=ax, publisher=GrassPublisher)

    integrate_with_qolo(
        position_start,
        # animator.initial_dynamics.evaluate,
        animator.avoider.evaluate,
        it_max=it_max,
        dt=0.02,
        ax=ax,
        attractor_position=dynamics.attractor_position,
    )
    breakpoint()


def plot_switching_path_following(n_grid=10, save_figure=False):
    x_lim = [-6.5, 8.5]
    y_lim = [-7.0, 7.0]

    # figsize = (4, 3.5)
    figsize = (8, 7.0)
    figtype = ".pdf"
    position_start = np.array([-5, -4.0])
    it_max = 2000

    dynamics = SwitchingDynamicsPathFollowing(
        RvizQoloAnimator.initial_dynamics.segments
    )
    conv_dynamics = SwitchingDynamics(RvizQoloAnimator.initial_dynamics.segments)

    # container = None
    container = random_placement_tables(
        x_lim=x_lim, y_lim=y_lim, start=position_start, goal=dynamics.attractor_position
    )

    animator = RvizQoloAnimator()
    animator.setup(x_lim=[-10, 10], y_lim=[-10, 10], container=container)
    plt.close("all")

    animator.avoider = RotationalAvoider(
        initial_dynamics=dynamics,
        convergence_system=conv_dynamics,
        obstacle_environment=animator.agent_container.get_obstacles(),
    )

    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        obstacle_container=[],
        # dynamics=dynamics.evaluate,
        dynamics=animator.avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        # attractor_position=dynamic.attractor_position,
        do_quiver=True,
        show_ticks=True,
    )
    plot_obstacles(
        ax=ax,
        obstacle_container=animator.agent_container.get_obstacles(),
        x_lim=x_lim,
        y_lim=y_lim,
        draw_reference=True,
    )
    plot_grass(ax=ax, publisher=GrassPublisher)

    integrate_with_qolo(
        position_start,
        # animator.initial_dynamics.evaluate,
        animator.avoider.evaluate,
        it_max=it_max,
        dt=0.02,
        ax=ax,
        attractor_position=dynamics.attractor_position,
    )
    breakpoint()


def get_fraction_outside(trajectory):
    grass_container = create_container_from_grass(publisher=GrassPublisher)

    collision_free = 0
    mean_sqrd_gamma = 0
    distances = np.zeros(trajectory.shape[1])
    for pp in range(trajectory.shape[1]):
        gamma = grass_container.get_minimum_gamma(trajectory[:, pp])
        if gamma >= 1:
            collision_free += 1

        mean_sqrd_gamma += gamma ** 2

        tmp_distances = np.zeros(len(grass_container))
        for ii, obs in enumerate(grass_container):
            tmp_distances[ii] = obs.get_distance_to_surface(
                trajectory[:, pp], in_global_frame=True
            )

        distances[pp] = np.min(tmp_distances)

        if gamma < 1:
            distances[pp] = distances[pp] * (-1)

    collision_ratio = 1.0 * collision_free / trajectory.shape[1]
    mean_sqrd_gamma = mean_sqrd_gamma / trajectory.shape[1]
    # return (collision_ratio, mean_sqrd_gamma)

    return (collision_ratio, np.mean(distances))


def get_distance(trajectory):
    local_dist = trajectory[:, :-1] - trajectory[:, 1:]
    return np.sum(np.linalg.norm(local_dist, axis=0))


def multi_test_switching_straight(x_lim, y_lim, n_runs, rndm_seed, start, it_max, dt):
    np.random.seed(rndm_seed)

    distances = np.zeros(n_runs)
    fraction_free = np.zeros(n_runs)
    converged = np.zeros(n_runs)
    gammas = np.zeros(n_runs)
    n_points = np.zeros(n_runs)

    for ii in range(n_runs):
        dynamics = SwitchingDynamics(RvizQoloAnimator.initial_dynamics.segments)

        # container = None
        container = random_placement_tables(
            x_lim=x_lim, y_lim=y_lim, start=start, goal=dynamics.attractor_position
        )

        animator = RvizQoloAnimator()
        animator.setup(x_lim=x_lim, y_lim=y_lim, container=container, do_plotting=0)

        animator.avoider = ModulationAvoider(
            initial_dynamics=dynamics,
            obstacle_environment=animator.agent_container.get_obstacles(),
        )

        trajectory = integrate_with_qolo(
            start,
            # animator.initial_dynamics.evaluate,
            animator.avoider.evaluate,
            it_max=it_max,
            dt=dt,
            ax=None,
            attractor_position=dynamics.attractor_position,
        )

        distances[ii] = get_distance(trajectory)
        fraction_free[ii], gammas[ii] = get_fraction_outside(trajectory)
        converged[ii] = np.allclose(
            trajectory[:, -1], dynamics.attractor_position, atol=1e-1
        )
        n_points[ii] = trajectory.shape[1]

    outfile = Path("media", f"wavy_path_switching_straight_randseed_{rndm_seed}.csv")
    np.savetxt(
        str(outfile),
        ammas=np.vstack((n_points, distances, fraction_free, converged, gammas)).T,
        header="n_points, distance, free_fraction, converged, gammas",
        delimiter=",",
    )


def multi_test_switching_pathfollowing(
    x_lim, y_lim, n_runs, rndm_seed, start, it_max, dt
):
    np.random.seed(rndm_seed)

    distances = np.zeros(n_runs)
    fraction_free = np.zeros(n_runs)
    converged = np.zeros(n_runs)
    gammas = np.zeros(n_runs)
    n_points = np.zeros(n_runs)

    for ii in range(n_runs):
        dynamics = SwitchingDynamicsPathFollowing(
            RvizQoloAnimator.initial_dynamics.segments
        )

        conv_dynamics = SwitchingDynamics(RvizQoloAnimator.initial_dynamics.segments)

        # container = None
        container = random_placement_tables(
            x_lim=x_lim, y_lim=y_lim, start=start, goal=dynamics.attractor_position
        )

        animator = RvizQoloAnimator()
        animator.setup(x_lim=x_lim, y_lim=y_lim, container=container, do_plotting=0)

        animator.avoider = RotationalAvoider(
            initial_dynamics=dynamics,
            convergence_system=conv_dynamics,
            obstacle_environment=animator.agent_container.get_obstacles(),
        )

        trajectory = integrate_with_qolo(
            start,
            # animator.initial_dynamics.evaluate,
            animator.avoider.evaluate,
            it_max=it_max,
            dt=dt,
            ax=None,
            attractor_position=dynamics.attractor_position,
        )

        distances[ii] = get_distance(trajectory)
        fraction_free[ii], gammas[ii] = get_fraction_outside(trajectory)
        converged[ii] = np.allclose(
            trajectory[:, -1], dynamics.attractor_position, atol=1e-1
        )
        n_points[ii] = trajectory.shape[1]

    outfile = Path("media", f"wavy_path_switching_path_randseed_{rndm_seed}.csv")
    np.savetxt(
        str(outfile),
        ammas=np.vstack((n_points, distances, fraction_free, converged, gammas)).T,
        header="n_points, distance, free_fraction, converged, gammas",
        delimiter=",",
    )


def get_nonlinear_global_trajectory(start, x_lim, y_lim, it_max, dt):
    # container = None
    container = random_placement_tables(
        x_lim=x_lim,
        y_lim=y_lim,
        start=start,
        goal=RvizQoloAnimator.initial_dynamics.attractor_position,
    )

    animator = RvizQoloAnimator()
    animator.setup(x_lim=x_lim, y_lim=y_lim, container=container, do_plotting=0)

    trajectory = integrate_with_qolo(
        start,
        # animator.initial_dynamics.evaluate,
        animator.avoider.evaluate_sequence,
        it_max=it_max,
        dt=dt,
        ax=None,
        attractor_position=animator.initial_dynamics.attractor_position,
    )
    return trajectory, animator.initial_dynamics.attractor_position


def multi_test_nonlinear_global(x_lim, y_lim, n_runs, rndm_seed, start, it_max, dt):
    np.random.seed(rndm_seed)

    distances = np.zeros(n_runs)
    fraction_free = np.zeros(n_runs)
    converged = np.zeros(n_runs)
    gammas = np.zeros(n_runs)
    n_points = np.zeros(n_runs)

    for ii in range(n_runs):
        trajectory, attractor = get_nonlinear_global_trajectory(
            start=start, x_lim=x_lim, y_lim=y_lim, it_max=it_max, dt=dt
        )

        distances[ii] = get_distance(trajectory)
        fraction_free[ii], gammas[ii] = get_fraction_outside(trajectory)
        converged[ii] = np.allclose(trajectory[:, -1], attractor, atol=1e-1)
        n_points[ii] = trajectory.shape[1]
        converged[ii]

    outfile = Path("media", f"wavy_path_global_nonlinear_randseed_{rndm_seed}.csv")
    np.savetxt(
        str(outfile),
        np.vstack((n_points, distances, fraction_free, converged, gammas)).T,
        header="n_points, distance, free_fraction, converged, gammas",
        delimiter=",",
    )


def run_comparison(n_runs=1):
    rndm_seed = 0
    x_lim = [-6.5, 8.5]
    y_lim = [-7.0, 7.0]
    it_max = 2000
    dt = 0.1
    start_position = np.array([-5, -4.0])

    # np.random.seed(rndm_seed)  # Do it here to, just to be sure..
    # multi_test_switching_straight(
    #     x_lim=x_lim,
    #     y_lim=y_lim,
    #     n_runs=n_runs,
    #     rndm_seed=rndm_seed,
    #     start=start_position,
    #     it_max=it_max,
    #     dt=dt,
    # )
    np.random.seed(rndm_seed)
    multi_test_nonlinear_global(
        x_lim=x_lim,
        y_lim=y_lim,
        n_runs=n_runs,
        rndm_seed=rndm_seed,
        start=start_position,
        it_max=it_max,
        dt=dt,
    )
    # np.random.seed(rndm_seed)
    # multi_test_switching_pathfollowing(
    #     x_lim=x_lim,
    #     y_lim=y_lim,
    #     n_runs=n_runs,
    #     rndm_seed=rndm_seed,
    #     start=start_position,
    #     it_max=it_max,
    #     dt=dt,
    # )


def check_convergence():
    x_lim = [-6.5, 8.5]
    y_lim = [-7.0, 7.0]
    it_max = 2000
    dt = 0.1
    start = np.array([-5, -4.0])

    for ii in range(50):
        np.random.seed(ii)
        trajectory, attractor = get_nonlinear_global_trajectory(
            start=start, x_lim=x_lim, y_lim=y_lim, it_max=it_max, dt=dt
        )

        converged = np.allclose(trajectory[:, -1], attractor, atol=1e-1)

        # print()
        # print(f"Converged: {converged} at it={ii}")
        # print()


def main():
    logging.basicConfig(level=logging.INFO)
    # This script is best to be run with the 'rviz_config'

    logging.info("Simulation started.")
    rclpy.init()
    try:
        main_wavy()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    logging.info("Simulation ended.")


if (__name__) == "__main__":
    # main()

    # plot_vectorfield_nonlinear_global(n_grid=10, save_figure=False)
    # plot_switching_linear_dynamics(n_grid=20)
    # plot_switching_path_following(n_grid=20)
    run_comparison(n_runs=100)
    # check_convergence()
