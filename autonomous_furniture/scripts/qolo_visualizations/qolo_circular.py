from dataclasses import dataclass, field
import logging

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster, TransformStamped

from vartools.states import ObjectPose
from vartools.dynamical_systems import QuadraticAxisConvergence, LinearSystem

from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import RotationalAvoider

from autonomous_furniture.message_generation import euler_to_quaternion
from autonomous_furniture.furniture_creators import create_hospital_bed

# from autonomous_furniture.rviz_animator import RvizSimulator


@dataclass(slots=True)
class Pose2D:
    position: np.ndarray = np.zeros(2)
    orientation: float = 0.0


@dataclass(slots=True)
class Twist2D:
    linear: np.ndarray = np.zeros(2)
    angular: float = 0.0


# @dataclass(slots=True)
class SimpleAgent:
    pose: Pose2D = field(default_factory=lambda: Pose2D())
    twist: Twist2D = field(default_factory=lambda: Twist2D())
    frame_id: str = ""


@dataclass(slots=True)
class RvizTable(SimpleAgent):
    pose: Pose2D = field(default_factory=lambda: Pose2D())
    twist: Twist2D = field(default_factory=lambda: Twist2D())
    it_count: int = 0
    _frame_id: str = ""

    def __post_init__(self):
        self._frame_id = f"qolo{str(self.it_count)}/base_link"

    def update_step(self, dt: float) -> None:
        self.pose.position = self.pose.position + self.twist.linear * dt
        self.pose.orientation = self.pose.orientation + self.twist.angular * dt

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
    pose: Pose2D = field(default_factory=lambda: Pose2D())
    twist: Twist2D = field(default_factory=lambda: Twist2D())
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

        transform_stamped.transform.rotation = euler_to_quaternion(
            0, 0, self.pose.orientation
        )  # rpy

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


def main(it_max=1000):
    start_position = np.array([0, 0])
    start_orientation = 0
    qolo = RvizQolo(pose=Pose2D(position=start_position, orientation=start_orientation))

    initial_dynamics = QuadraticAxisConvergence(
        stretching_factor=3,
        maximum_velocity=1.0,
        dimension=2,
        attractor_position=np.array([8, 0]),
    )
    convergence_dynamics = LinearSystem(
        attractor_position=initial_dynamics.attractor_position
    )

    obstacle_environment = RotationContainer()
    obstacle_environment.add_obstacle(
        create_hospital_bed(
            start_pose=ObjectPose(position=np.array([0, 0]), orientation=0),
            margin_absolut=qolo.required_margin,
        )
    )

    # obstacle_environment.set_convergence_directions(initial_dynamics)

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_environment,
        convergence_system=convergence_dynamics,
    )

    for ii in range(it_max):
        qolo.twist.linear = my_avoider.evaluate(qolo.pose.position)

        qolo.update_step()


if (__name__) == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

    logging.info("Simulation ended.")
