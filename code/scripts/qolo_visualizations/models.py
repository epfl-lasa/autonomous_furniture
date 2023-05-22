import copy
from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from vartools.states import Pose, Twist

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers import BaseContainer, ObstacleContainer

# from autonomous_furniture.rviz_animator import RvizSimulator
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster, TransformStamped

from autonomous_furniture.launch_generator import create_table_node
from autonomous_furniture.message_generation import euler_to_quaternion


@dataclass(slots=True)
class SimpleAgent:
    shapes: list[Obstacle]

    pose: Pose = field(default_factory=lambda: Pose.create_trivial(2))
    twist: Twist = field(default_factory=lambda: Twist.create_trivial(2))
    name: str = ""

    it_count: ClassVar[int] = 0

    def __post_init__(self) -> None:
        # They are all published as 'robots'
        self.name = f"furniture{str(self.it_count)}"
        RvizTable.it_count += 1

    @property
    def frame_id(self):
        return self.name + "/base_link"


def update_shapes_of_agent(agent):
    for ii, obs in enumerate(agent.shapes):
        obs.pose = agent.pose.transform_pose_from_relative(
            copy.deepcopy(agent.local_poses[ii])
        )


@dataclass(slots=True)
class RvizTable(SimpleAgent):
    shapes: list[Obstacle] = field(
        default_factory=lambda: [
            Cuboid(
                pose=Pose.create_trivial(2),
                axes_length=np.array([1.5, 0.75]),
                tail_effect=False,
                distance_scaling=1.5,
            )
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

    name: str = "qolo_human"

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

    @property
    def n_obstacles(self) -> int:
        return len(self._agent_list)

    def append(self, agent: SimpleAgent) -> None:
        self._agent_list.append(agent)

    def get_single_obstacle(self, ind: int) -> Obstacle:
        if len(self._agent_list[ind].shapes) > 1:
            raise NotImplementedError()

        return self._agent_list[ind].shapes[0]

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
            self.broadcaster.sendTransform(self.transform_stamped)
