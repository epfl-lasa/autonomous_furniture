from typing import ClassVar, Optional
from dataclasses import dataclass, field
import logging

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster, TransformStamped

from vartools.states import Pose, Twist
from vartools.dynamical_systems import QuadraticAxisConvergence, LinearSystem

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.containers import BaseContainer, ObstacleContainer

from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import RotationalAvoider

from autonomous_furniture.message_generation import euler_to_quaternion
from autonomous_furniture.furniture_creators import create_hospital_bed

# from autonomous_furniture.rviz_animator import RvizSimulator


# @dataclass(slots=True)
class SimpleAgent:
    pose: Pose = field(default_factory=lambda: Pose.create_trivial())
    twist: Twist = field(default_factory=lambda: Twist.create_trivial())
    shapes: list[Obstacle]
    frame_id: str = ""

    def update_obstacle_shapes(self):
        for obs in shapes:
            obs.pose = self.pose.transform_pose_from_relative(obs.pose)


@dataclass(slots=True)
class AgentContainer:
    _agent_list: list[SimpleAgent] = field(default_factory=list)

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


@dataclass(slots=True)
class RvizTable(SimpleAgent):
    pose: Pose = field(default_factory=lambda: Pose.create_trivial(2))
    twist: Twist = field(default_factory=lambda: Twist.create_trivial(2))
    _frame_id: str = ""

    it_count: ClassVar[int] = 0
    shapes: list[Obstacle] = field(
        default_factory=lambda: [
            Cuboid(pose=Pose.create_trivial(2), axes_length=np.array([2.0, 1.0]))
        ]
    )
    local_poses: list[Obstacle] = field(
        default_factory=lambda: [Pose.create_trivial(2)]
    )

    level_list: Optional[list[int]] = None

    def __post_init__(self) -> None:
        self._frame_id = f"qolo{str(self.it_count)}/base_link"
        RvizTable.it_count += 1

    def update_step(self, dt: float) -> None:
        self.pose.position = self.pose.position + self.twist.linear * dt
        self.pose.orientation = self.pose.orientation + self.twist.angular * dt

        self.update_obstacle_shapes()

    def update_transform(
        self, transform_stamped: TransformBroadcaster
    ) -> TransformBroadcaster:
        transform_stamped.child_frame_id = self._frame_id
        transform_stamped.transform.translation.x = self.pose.position[0] + 0.2
        transform_stamped.transform.translation.y = self.pose.position[1]
        transform_stamped.transform.translation.z = 0.2

        transform_stamped.transform.rotation = euler_to_quaternion(
            0, 0, self.pose.orientation
        )


@dataclass(slots=True)
class RvizQolo:
    pose: Pose = field(default_factory=lambda: Pose.create_trivial(2))
    twist: Twist = field(default_factory=lambda: Twist.create_trivial(2))
    it_count: int = 0
    _frame_id: str = ""

    required_margin: int = 0.5

    def __post_init__(self):
        self._frame_id = f"qolo{str(self.it_count)}/base_link"

    def update_step(self, dt: float) -> None:
        self.pose.position = self.pose.position + self.twist.linear * dt
        self.pose.orientation = np.arctan2(self.twist.linear[1], self.twist.linear[0])

    def update_transform(
        self, transform_stamped: TransformBroadcaster
    ) -> TransformBroadcaster:
        transform_stamped.child_frame_id = self._frame_id
        transform_stamped.transform.translation.x = self.pose.position[0]
        transform_stamped.transform.translation.y = self.pose.position[1]
        transform_stamped.transform.translation.z = 0.2

        # roll-pitch-yaw
        transform_stamped.transform.rotation = euler_to_quaternion(
            0, 0, self.pose.orientation
        )

        return transform_stamped


# @dataclass(slots=True)c
class RvizQoloAnimator(Node):
    def __init__(
        self,
        robot,
        avoider,
        it_max: int,
        dt_sleep: float = 0.05,
        dt_simulation: float = 0.01,
    ) -> None:
        self.animation_paused = False

        self.it_max = it_max
        self.dt_sleep = dt_sleep
        self.dt_simulation = dt_simulation

        super().__init__("transform_publisher")

        self.period = self.dt_simulation

        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.node_name = self.get_name()
        self.transform_stamped = TransformStamped()
        self.transform_stamped.header.frame_id = "world"

        self.ii = 0

        self._pause = False
        self.timer = self.create_timer(self.period, self.update_step)

        self.avoider = avoider
        self.robot = robot

    def update_step(self) -> None:
        self.ii += 1

        self.robot.twist.linear = self.avoider.evaluate(self.robot.position)
        self.robot.update_step(self.dt_simulation)

        self.transform_stamped.header.stamp = self.get_clock().now().to_msg()

        # Only avoid QOLO
        self.transform_stamped = self.robot.update_transform(self.transform_stamped)
        self.broadcaster.sendTransform(self.transform_stamped)


def main(it_max: int = 1000, delta_time: float = 0.1):
    start_position = np.array([3, 0])
    start_orientation = 0
    qolo = RvizQolo(pose=Pose(position=start_position, orientation=start_orientation))

    initial_dynamics = QuadraticAxisConvergence(
        stretching_factor=3,
        maximum_velocity=1.0,
        dimension=2,
        attractor_position=np.array([8, 0]),
    )
    convergence_dynamics = LinearSystem(
        attractor_position=initial_dynamics.attractor_position
    )

    agent_container = AgentContainer()
    agent_container.append(RvizTable(pose=Pose.create_trivial(2)))
    # obstacle_environment.set_convergence_directions(initial_dynamics)

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        convergence_system=convergence_dynamics,
    )

    for ii in range(it_max):
        qolo.twist.linear = my_avoider.avoid(
            position=qolo.pose.position,
            obstacle_list=agent_container.get_obstacles(
                desired_margin=qolo.required_margin
            ),
        )
        qolo.update_step(dt=delta_time)


if (__name__) == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

    logging.info("Simulation ended.")
