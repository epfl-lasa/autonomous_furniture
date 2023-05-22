import copy
import time
import logging
import multiprocessing
from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from vartools.states import Pose, Twist
from vartools.dynamical_systems import QuadraticAxisConvergence, LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers import BaseContainer, ObstacleContainer
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.arch_obstacle import MultiObstacleContainer, BlockArchObstacle

# from nonlinear_avoidance.rotation_container import RotationContainer
# from nonlinear_avoidance.avoidance import RotationalAvoider
# from autonomous_furniture.rviz_animator import RvizSimulator
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

from autonomous_furniture.message_generation import euler_to_quaternion

# Relative Package
from qolo_along_road import AgentTransformBroadCaster
from qolo_along_road import RvizQolo
from qolo_along_road import update_shapes_of_agent

# from qolo_along_road import RvizQoloAnimator


class WallPublisher(Node):
    def __init__(self):
        super().__init__("environment_publisher")
        self.markers_array = MarkerArray()

        self.publisher_ = self.create_publisher(MarkerArray, "environment", 3)
        self.wall_it = 0

        self.markers_array.markers.append(self.create_ground())

        self.publisher_.publish(self.markers_array)

    def add_nodes_from_multibsticle_container(self, container: MultiObstacleContainer):
        for obstacle_tree in container._obstacle_list:
            for component in obstacle_tree._obstacle_list:
                if not isinstance(component, Cuboid):
                    raise ValueError()

                self.markers_array.markers.append(
                    self.create_wall_from_obstacle(
                        cuboid=component, ns=f"wall{self.wall_it}"
                    )
                )
                self.wall_it += 1
                print("Done another")

        self.publisher_.publish(self.markers_array)

    def create_wall_from_obstacle(self, cuboid: Cuboid, ns: str) -> Marker:
        marker = Marker()
        # def publish_cube(self):
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 0
        marker.type = 1  # is cube
        # marker.action = visualization_msgs::Marker::ADD
        marker.pose.position.x = cuboid.pose.position[0]
        marker.pose.position.y = cuboid.pose.position[1]
        marker.pose.position.z = 0.0
        quat = Rotation.from_euler("z", cuboid.orientation).as_quat()

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker.scale.x = cuboid.axes_length[0]
        marker.scale.y = cuboid.axes_length[1]
        marker.scale.z = 2.0
        # 170, 74, 68. [brick color]
        marker.color.a = 1.0
        marker.color.r = 170 / 256.0
        marker.color.g = 75 / 256.0
        marker.color.b = 68 / 256.0
        return marker

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
        marker.scale.x = 10.0
        marker.scale.y = 10.0
        marker.scale.z = 0.01
        # 255, 87, 51.
        marker.color.a = 1.0
        marker.color.r = 150 / 256.0
        marker.color.g = 150 / 256.0
        marker.color.b = 150 / 256.0
        return marker


class QoloWallsAnimator(Animator):
    def setup(
        self,
        broadcaster: Optional[AgentTransformBroadCaster] = None,
        do_plotting: bool = True,
        x_lim=[-5, 5],
        y_lim=[-5, 5],
    ) -> None:
        self.x_lim = x_lim
        self.y_lim = y_lim

        start_position = np.array([-4.4, 1.5])
        start_orientation = 0
        self.robot = RvizQolo(
            pose=Pose(position=start_position, orientation=start_orientation)
        )

        attractor = np.array([4.0, -3])
        # initial_dynamics = QuadraticAxisConvergence(
        #     stretching_factor=10,
        #     maximum_velocity=1.0,
        #     dimension=2,
        #     attractor_position=attractor,
        # )
        initial_dynamics = LinearSystem(
            attractor_position=attractor,
            maximum_velocity=1.0,
        )

        self.container = MultiObstacleContainer()
        self.container.append(
            BlockArchObstacle(
                wall_width=0.4,
                axes_length=np.array([4.5, 6.5]),
                pose=Pose(np.array([-1.5, -3.5]), orientation=90 * np.pi / 180.0),
                margin_absolut=self.robot.required_margin,
            )
        )

        self.container.append(
            BlockArchObstacle(
                wall_width=0.4,
                axes_length=np.array([4.5, 6.0]),
                pose=Pose(np.array([1.5, 3.0]), orientation=-90 * np.pi / 180.0),
                margin_absolut=self.robot.required_margin,
            )
        )

        self.avoider = MultiObstacleAvoider(
            obstacle_container=self.container,
            initial_dynamics=initial_dynamics,
            # convergence_dynamics=rotation_projector,
            create_convergence_dynamics=True,
        )

        self.broadcaster = broadcaster

        self.do_plotting = do_plotting
        if do_plotting:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = None

        # Initial update of transform
        update_shapes_of_agent(self.robot)
        # for agent in self.agent_container:
        #     update_shapes_of_agent(agent)

    def update_step(self, ii):
        print("ii", ii)
        self.robot.twist.linear = self.avoider.evaluate(
            position=self.robot.pose.position,
            # obstacle_list=self.container,
        )
        self.robot.update_step(dt=self.dt_simulation)
        update_shapes_of_agent(self.robot)
        if self.broadcaster is not None:
            self.broadcaster.broadcast([self.robot])
            # self.broadcaster.broadcast(se)

        if not self.do_plotting:
            return

        self.ax.clear()
        plot_obstacles(ax=self.ax, obstacle_container=self.robot.shapes, noTicks=True)
        for obstacle_tree in self.container:
            plot_obstacles(
                ax=self.ax,
                obstacle_container=obstacle_tree._obstacle_list,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                noTicks=True,
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


def main(
    it_max: int = 1000,
    delta_time: float = 0.1,
    do_ros: bool = True,
    # do_plotting: bool = False,
    do_plotting: bool = True,
):
    if do_ros:
        broadcaster = AgentTransformBroadCaster()
    else:
        broadcaster = None

    animator = QoloWallsAnimator(
        it_max=it_max,
        dt_simulation=delta_time,
        dt_sleep=delta_time,
    )
    animator.setup(
        broadcaster=broadcaster, do_plotting=do_plotting, x_lim=[-5, 5], y_lim=[-5, 5]
    )
    publisher = WallPublisher()
    publisher.add_nodes_from_multibsticle_container(animator.container)

    # Create launch rviz
    print("Done Launching")
    if do_plotting:
        animator.run()
    else:
        for ii in range(it_max):
            animator.update_step(ii)
            time.sleep(delta_time)

    print("End of script")


if (__name__) == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Simulation started.")
    rclpy.init()
    try:
        main()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

    logging.info("Simulation ended.")
