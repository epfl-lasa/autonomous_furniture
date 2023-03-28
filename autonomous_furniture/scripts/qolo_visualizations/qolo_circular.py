from dataclasses import dataclass, field

import numpy as np

from tf2_ros import TransformBroadcaster, TransformStamped
from autonomous_furniture.message_generation import euler_to_quaternion


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
    pose: Pose2D = field(default_factory=labmda: Pose2D())
    twist: Twist2D = field(default_factory=labmda: Twist2D())

    frame_id: str = ""


class RvizTable:
    def __post_init__(self, it_count: str = ""):
        self.frame_id = f"table{it_count}/base_link"
        
    def update_step(self, dt: float) -> None:
        self.pose.position = self.pose.position + self.twist.linear * dt
        self.pose.orientation = self.pose.orientation + self.twist.angular * dt

    def update_transform(self, transform_stamped: TransformBroadcaster) -> TransformBroadcaster:
        transform_stamped.child_frame_id = self.frame_id
        transform_stamped.transform.translation.x = self.pose.position[0] + 0.2
        transform_stamped.transform.translation.y = self.pose.position[1]
        transform_stamped.transform.translation.z = 0.2

        transform_stamped.transform.rotation = euler_to_quaternion(
            0, 0, self.pose.orientation
            )

class RvizQolo(SimpleAgent):
    def __post_init__(self, it_count: str = ""):
        self.frame_id = f"qolo{it_count}/base_link"
        
    def update_step(self, dt: float) -> None:
        self.pose.position = self.pose.position + self.twist.linear * dt
        self.pose.orientation = np.arctan2(self.twist.linear[1], self.twist.linear[0])

    def update_transform(self, transform_stamped: TransformBroadcaster) -> TransformBroadcaster:
        transform_stamped.child_frame_id = self.frame_id
        transform_stamped.transform.translation.x = self.pose.position[0]
        transform_stamped.transform.translation.y = self.pose.position[1]
        transform_stamped.transform.translation.z = 0.2

        transform_stamped.transform.rotation = euler_to_quaternion(
                0, 0, pose.orientation
            )  # rpy

        return transform_stamped


# @dataclass(slots=True)c
class RvizQoloAnimator(Node):
    def __init__(
        self, it_max: int, dt_sleep: float = 0.05, dt_simulation: float = 0.01
    ) -> None:
        self.animation_paused = False

        self.it_max = it_max
        self.dt_sleep = dt_sleep
        self.dt_simulation = dt_simulation
        
        super().__init__("qolo_publisher")

        self.period = self.dt_simulation

        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.node_name = self.get_name()
        self.transform_stamped = TransformStamped()
        self.transform_stamped.header.frame_id = "world"

        self.ii = 0
        
        self._pause = False
        self.timer = self.create_timer(self.period, self.update_step)

    def update_step(self) -> None:
        self.ii += 1

        self.transform_stamped.header.stamp = self.get_clock().now().to_msg()

        # Only avoid QOLO
        self.transform_stamped = self.qolo.update_transform(self.transform_stamped)
        self.broadcaster.sendTransform(self.transform_stamped)

        # Only avoid QOLO
        self.transform_stamped = self.qolo.update_transform(self.transform_stamped)
        self.broadcaster.sendTransform(self.transform_stamped)

    

def main():
    import rclpy
    from rclpy.node import Node
    from autonomous_furniture.rviz_animator import RvizSimulator

    start_position = np.array([0, 0])
    start_orientation = 0
    qolo = RvizQolo(pose=Pose2D(position=start_position, orientation=start_orientation))
    

    for ii in range(it_max):
        pass

    

if (__name__) == "__main__":
    main()
