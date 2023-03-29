import copy
import logging
import multiprocessing
from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from vartools.states import Pose, Twist
from vartools.dynamical_systems import QuadraticAxisConvergence, LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers import BaseContainer, ObstacleContainer

from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import RotationalAvoider

# from autonomous_furniture.rviz_animator import RvizSimulator
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster, TransformStamped
import launch

from autonomous_furniture.launch_generator import create_bed_node
from autonomous_furniture.launch_generator import create_table_node
from autonomous_furniture.launch_generator import generate_launch_description
from autonomous_furniture.message_generation import euler_to_quaternion


@dataclass(slots=True)
class SimpleAgent:
    shapes: list[Obstacle]

    pose: Pose = field(default_factory=lambda: Pose.create_trivial(2))
    twist: Twist = field(default_factory=lambda: Twist.create_trivial(2))
    name: str = ""

    it_count: ClassVar[int] = 0

    @property
    def frame_id(self):
        return self.name + "/base_link"

    def __post_init__(self) -> None:
        # They are all published as 'robots'
        self.name = f"robot{str(self.it_count)}"
        RvizTable.it_count += 1


def update_shapes_of_agent(agent):
    for ii, obs in enumerate(agent.shapes):
        obs.pose = agent.pose.transform_pose_from_relative(
            copy.deepcopy(agent.local_poses[ii])
        )


@dataclass(slots=True)
class RvizTable(SimpleAgent):
    shapes: list[Obstacle] = field(
        default_factory=lambda: [
            Cuboid(pose=Pose.create_trivial(2), axes_length=np.array([2.0, 1.0]))
        ]
    )
    local_poses: list[Obstacle] = field(
        default_factory=lambda: [Pose.create_trivial(2)]
    )

    level_list: Optional[list[int]] = None

    def update_step(self, dt: float) -> None:
        self.pose.position = self.pose.position + self.twist.linear * dt
        self.pose.orientation = self.pose.orientation + self.twist.angular * dt

        self.update_obstacle_shapes()

    def get_node(self) -> Node:
        return create_table_node(name=self.name)

    def update_transform(
        self, transform_stamped: TransformBroadcaster
    ) -> TransformBroadcaster:
        transform_stamped.child_frame_id = self.frame_id
        transform_stamped.transform.translation.x = self.pose.position[0] + 0.2
        transform_stamped.transform.translation.y = self.pose.position[1]
        transform_stamped.transform.translation.z = 0.2

        transform_stamped.transform.rotation = euler_to_quaternion(
            0, 0, self.pose.orientation
        )

        return transform_stamped


@dataclass(slots=True)
class RvizQolo:
    pose: Pose = field(default_factory=lambda: Pose.create_trivial(2))
    twist: Twist = field(default_factory=lambda: Twist.create_trivial(2))

    shapes: list[Obstacle] = field(
        default_factory=lambda: [
            Ellipse(pose=Pose.create_trivial(2), axes_length=np.array([0.9, 0.9]))
        ]
    )
    local_poses: list[Obstacle] = field(
        default_factory=lambda: [Pose.create_trivial(2)]
    )
    required_margin: int = 0.5

    name: str = "qolo"

    def update_step(self, dt: float) -> None:
        self.pose.position = self.pose.position + self.twist.linear * dt
        self.pose.orientation = np.arctan2(self.twist.linear[1], self.twist.linear[0])

    @property
    def frame_id(self):
        return self.name + "/base_link"

    def update_transform(
        self, transform_stamped: TransformBroadcaster
    ) -> TransformBroadcaster:
        transform_stamped.child_frame_id = self.frame_id
        transform_stamped.transform.translation.x = self.pose.position[0]
        transform_stamped.transform.translation.y = self.pose.position[1]
        transform_stamped.transform.translation.z = 0.2

        # roll-pitch-yaw
        transform_stamped.transform.rotation = euler_to_quaternion(
            0, 0, self.pose.orientation
        )

        return transform_stamped


@dataclass(slots=True)
class AgentContainer:
    _agent_list: list[SimpleAgent] = field(default_factory=list)

    def __iter__(self):
        return iter(self._agent_list)

    def append(self, agent: SimpleAgent) -> None:
        self._agent_list.append(agent)

    def get_obstacles(
        self,
        excluding_agents: list[SimpleAgent] = [],
        level: Optional[int] = None,
        desired_margin: Optional[float] = None,
    ) -> BaseContainer:
        """Returns an ObstacleContainer based on the obstacles, i.e., the avoidance-shapes."""
        # Note that the (mutable) default argument is never changed.

        container_ = ObstacleContainer()
        for agent in self._agent_list:
            if agent in excluding_agents:
                continue

            for obs in agent.shapes:
                if desired_margin is not None:
                    obs.margin_absolut = desired_margin
                if level is None:
                    container_.append(obs)
                    continue

                # TODO: only take specific level [future implementation]
                pass

        return container_


class AgentTransformBroadCaster(Node):
    def __init__(self, agent_container=None, period: float = 0.1) -> None:
        self.period = period

        super().__init__("transform_publisher")

        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.transform_stamped = TransformStamped()
        self.transform_stamped.header.frame_id = "world"

        self.agent_container = agent_container

        # The node could run autonomously
        self.it_counter = 0
        if self.agent_container is not None:
            self.timer = self.create_timer(self.period, self.update_step)

    def update_step(self) -> None:
        self.it_counter += 1  # Counter
        self.broadcast(self.agent_container)

    def broadcast(self, agent_container):
        self.transform_stamped.header.stamp = self.get_clock().now().to_msg()

        for agent in agent_container:
            # Only avoid QOLO
            self.transform_stamped = agent.update_transform(self.transform_stamped)
            try:
                self.broadcaster.sendTransform(self.transform_stamped)
            except:
                breakpoint()


class RvizQoloAnimator(Animator):
    def setup(
        self,
        broadcaster: Optional[AgentTransformBroadCaster] = None,
        do_plotting: bool = True,
        x_lim=[-10, 10],
        y_lim=[-10, 10],
    ) -> None:
        self.x_lim = x_lim
        self.y_lim = y_lim

        start_position = np.array([4, 4])
        start_orientation = 0
        self.robot = RvizQolo(
            pose=Pose(position=start_position, orientation=start_orientation)
        )

        initial_dynamics = QuadraticAxisConvergence(
            stretching_factor=3,
            maximum_velocity=1.0,
            dimension=2,
            attractor_position=np.array([8, 0]),
        )
        convergence_dynamics = LinearSystem(
            attractor_position=initial_dynamics.attractor_position
        )

        self.agent_container = AgentContainer()
        self.agent_container.append(RvizTable(pose=Pose.create_trivial(2)))
        self.avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            convergence_system=convergence_dynamics,
        )

        self.broadcaster = broadcaster

        self.do_plotting = do_plotting
        if do_plotting:
            self.fig, self.ax = plt.subplots()

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


def main(
    it_max: int = 1000,
    delta_time: float = 0.1,
    do_ros: bool = True,
    do_plotting: bool = True,
):

    if do_ros:
        broadcaster = AgentTransformBroadCaster()
    else:
        broadcaster = None
    animator = RvizQoloAnimator(
        it_max=it_max,
        dt_simulation=delta_time,
        dt_sleep=delta_time,
    )
    animator.setup(
        broadcaster=broadcaster,
        do_plotting=True,
        x_lim=[-10, 10],
        y_lim=[-10, 10],
    )

    # Create launch rviz
    nodes = []
    for agent in animator.agent_container:
        nodes.append(agent.get_node())

    print("Done Launching")
    # if do_ros:
    # ros_animator = RosAnimatorNode(animator)
    # process = multiprocessing.Process(target=animator.spin).start()
    animator.run()
    # process = multiprocessing.Process(target=animator.run).start()
    print("End of script")
    if True:
        return

    launch_description = generate_launch_description(nodes, create_rviz=False)
    launch_service = launch.LaunchService()
    launch_service.include_launch_description(launch_description)
    process = multiprocessing.Process(target=launch_service.run).start()


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
