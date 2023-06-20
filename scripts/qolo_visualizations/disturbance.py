import numpy as np

from pynput import mouse

from matplotlib.backend_bases import MouseButton

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

from vartools.linalg import get_rotation_between_vectors


class VelocityPublisher(Node):
    def __init__(self, name: str, color: tuple[float]):
        # self.name = name
        super().__init__(f"velocity_publisher_{name}")

        self.marker = self.create_arrow(color)
        self.publisher = self.create_publisher(Marker, f"velocity/{name}", 3)

    def create_arrow(self, color: tuple[float], ns="velocity"):
        marker = Marker()

        marker.header.frame_id = "world"
        marker.ns = ns
        marker.id = 0
        marker.type = 0  # Arrow
        # marker.action = visualization_msgs::Marker::ADD

        marker.scale.x = 0.0
        marker.scale.y = 0.0
        marker.scale.z = 0.0

        # 170, 74, 68. [brick color]

        marker.color.a = 1.0
        marker.color.r = color[0] / 256.0
        marker.color.g = color[1] / 256.0
        marker.color.b = color[2] / 256.0

        return marker

    def publish(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Update the marker with 2d-position and 2d-velocity."""
        self.marker.header.stamp = self.get_clock().now().to_msg()

        self.marker.pose.position.x = position[0]
        self.marker.pose.position.y = position[1]
        self.marker.pose.position.z = 0.5

        norm = np.linalg.norm(velocity) * 0.3
        self.marker.scale.x = norm
        self.marker.scale.y = norm * 0.2
        self.marker.scale.z = norm * 0.2

        orientation = get_rotation_between_vectors([1, 0, 0], np.append(velocity, 0))

        quat = orientation.as_quat()
        self.marker.pose.orientation.x = quat[0]
        self.marker.pose.orientation.y = quat[1]
        self.marker.pose.orientation.z = quat[2]
        self.marker.pose.orientation.w = quat[3]

        self.publisher.publish(self.marker)


class MouseDisturbanceWithFigure(Node):
    def __init__(self, fig):
        self.dimension = 2

        self.clicked = False
        self.click_position: np.ndarray
        self.last_position: np.ndarray

        self.fig = fig

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    @property
    def click_offset(self):
        if not self.clicked:
            return np.zeros(self.dimension)

        return self.last_position - self.click_position

    def on_click(self, event=None) -> None:
        self.clicked = True
        self.click_position = np.array([event.x, event.y])
        self.last_position = np.array([event.x, event.y])

    def on_move(self, event):
        self.last_position = np.array([event.x, event.y])

    def on_release(self, event):
        self.clicked = False


class MouseDisturbance(Node):
    def __init__(self):
        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            # on_scroll=on_scroll
        )
        self.listener.start()

        self.dimension = 2

        self.clicked = False
        self.click_position: np.ndarray
        self.last_position: np.ndarray

        self.listener = mouse.Listener(
            on_click=self.on_click,
            # on_scroll=on_scroll
        )

    @property
    def click_offset(self):
        if not self.clicked:
            return np.zeros(self.dimension)

        return self.last_position - self.click_position

    def on_click(self, xx, yy, button, pressed):
        print("Click event detected.")

        if not pressed:
            self.clicked = False
            return

        self.clicked = True
        self.click_position = np.array([xx, yy])

    def on_move(self, xx, yy):
        self.last_position = np.array([xx, yy])
