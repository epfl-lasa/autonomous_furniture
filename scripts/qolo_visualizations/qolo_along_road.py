import time
import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.avoidance import RotationalAvoider

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

    def __init__(self, x_lim=[-5, 5], y_lim=[-5, 5]):
        super().__init__("environment_publisher")
        self.markers_array = MarkerArray()

        self.publisher_ = self.create_publisher(MarkerArray, "environment", 3)
        self.markers_array.markers.append(self.create_ground())
        self.markers_array.markers.append(
            self.create_grass(x_pos=0.0, y_pos=4.0, ns="grass0")
        )
        self.markers_array.markers.append(
            self.create_grass(x_pos=0.0, y_pos=-4.0, ns="grass1")
        )

        self.place_lines(n_lines=5)
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
        marker.scale.x = 10.0
        marker.scale.y = 10.0
        marker.scale.z = 0.01
        # 255, 87, 51.
        marker.color.a = 1.0
        marker.color.r = 211 / 256.0
        marker.color.g = 211 / 256.0
        marker.color.b = 211 / 256.0
        return marker

    def create_grass(self, x_pos=0.0, y_pos=-4.0, ns="grass"):
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
        marker.scale.x = 10.0
        marker.scale.y = 3.0
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

        attractor = np.array([4.5, 0])

        initial_dynamics = LinearSystem(
            attractor_position=attractor, maximum_velocity=1.0
        )
        convergence_dynamics = LinearSystem(
            attractor_position=initial_dynamics.attractor_position
        )

        self.agent_container = AgentContainer()
        self.agent_container.append(
            RvizTable(pose=Pose(position=[-2.5, 1], orientation=-0.1 * np.pi))
        )
        self.agent_container.append(
            RvizTable(pose=Pose(position=[-2.0, -1.5], orientation=np.pi / 2))
        )
        self.agent_container.append(
            RvizTable(pose=Pose(position=[0.6, -1.5], orientation=0.2 * np.pi))
        )
        self.agent_container.append(
            RvizTable(pose=Pose(position=[2.5, -0.3], orientation=0.4 * np.pi))
        )
        self.agent_container.append(
            RvizTable(pose=Pose(position=[0.3, 1.3], orientation=-0.3 * np.pi))
        )

        self.avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            convergence_system=convergence_dynamics,
        )

        self.broadcaster = broadcaster

        self.do_plotting = do_plotting
        if do_plotting:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = None

        # Initial update of transform
        update_shapes_of_agent(self.robot)
        for agent in self.agent_container:
            update_shapes_of_agent(agent)

    def update_step(self, ii):
        print("ii", ii)
        self.robot.twist.linear = self.avoider.avoid(
            position=self.robot.pose.position,
            obstacle_list=self.agent_container.get_obstacles(
                desired_margin=self.robot.required_margin
            ),
        )
        self.robot.update_step(dt=self.dt_simulation)
        update_shapes_of_agent(self.robot)
        if self.broadcaster is not None:
            self.broadcaster.broadcast([self.robot])
            self.broadcaster.broadcast(self.agent_container)

        if not self.do_plotting:
            return

        self.ax.clear()
        plot_obstacles(ax=self.ax, obstacle_container=self.robot.shapes, noTicks=True)
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.agent_container.get_obstacles(
                desired_margin=self.robot.required_margin
            ),
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
        broadcaster=broadcaster, do_plotting=do_plotting, x_lim=[-5, 5], y_lim=[-5, 5]
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


if (__name__) == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Simulation started.")
    rclpy.init()
    try:
        main_wavy()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

    logging.info("Simulation ended.")
