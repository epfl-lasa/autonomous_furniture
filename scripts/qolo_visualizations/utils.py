from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import numpy as np


class TrajectoryPublisher(Node):
    def __init__(self, animator, avoid_functor=None, atol=1e-3):
        super().__init__("trajectory_publisher")

        dimension = 2
        trajectory = np.zeros((dimension, animator.it_max + 1))
        trajectory[:, 0] = animator.start_position

        if avoid_functor is None:
            avoid_functor = animator.avoider.evaluate_sequence

        self.path = Path()
        for ii in range(animator.it_max):
            velocity = avoid_functor(position=trajectory[:, ii])
            if np.linalg.norm(velocity) < atol:
                trajectory = trajectory[:, : ii + 1]
                break

            trajectory[:, ii + 1] = (
                velocity * animator.dt_simulation + trajectory[:, ii]
            )

            if not ii % 20:
                print(f"Preparation Loop {ii}")

        stamp = self.get_clock().now().to_msg()
        frame_id = "world"
        for ii in range(trajectory.shape[1]):
            self.path.poses.append(PoseStamped())
            self.path.poses[ii].pose.position.x = trajectory[0, ii]
            self.path.poses[ii].pose.position.y = trajectory[1, ii]
            self.path.poses[ii].pose.position.z = 0.0
            # self.path.poses[ii].pose.orientation.x = 0.0
            # self.path.poses[ii].pose.orientation.y = 0.0
            # self.path.poses[ii].pose.orientation.z = 0.0
            # self.path.poses[ii].pose.orientation.w = 0.0
            self.path.poses[ii].header.frame_id = frame_id
            self.path.poses[ii].header.stamp = stamp

        self.path.header.frame_id = frame_id
        self.path.header.stamp = stamp

        # Publish
        self.path_pub = self.create_publisher(Path, "/qolo/path", 10)
        self.path_pub.publish(self.path)

        print(trajectory)
