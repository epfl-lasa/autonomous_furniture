from math import sin, cos, pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformBroadcaster, TransformStamped

DISPLACEMENT = 0.01


class StatePublisher(Node):
    def __init__(self):
        rclpy.init()
        super().__init__("state_publisher")

        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()

        degree = pi / 180.0
        loop_rate = self.create_rate(30)

        # robot state, this should be changed TODO
        angle = 0.0
        pos_xy = [-3.0, 0.0]
        prev_dir_x = 0
        prev_dir_y = 0

        # message declarations
        odom_trans = TransformStamped()
        odom_trans.header.frame_id = "odom"
        odom_trans.child_frame_id = "qolo_tf"  # may need to change this TODO

        try:
            while rclpy.ok():
                rclpy.spin_once(self)

                # update joint_state
                now = self.get_clock().now()

                # update transform
                # (moving in a circle with radius = 2)
                odom_trans.header.stamp = now.to_msg()
                odom_trans.transform.translation.x = pos_xy[0]
                odom_trans.transform.translation.y = pos_xy[1]
                odom_trans.transform.translation.z = 0.2
                odom_trans.transform.rotation = euler_to_quaternion(0, 0, 0)  # rpy

                # send the joint state and transform
                self.broadcaster.sendTransform(odom_trans)

                # create new robot state, TODO
                angle += degree / 4

                # if pos_xy[0] > -1.0:
                #     pos_xy[0] = -3.0
                # else:
                #     pos_xy[0] += DISPLACEMENT

                pos_xy[0] += DISPLACEMENT

                # if pos_xy[0] < -4.0:
                #     pos_xy[0] += DISPLACEMENT
                #     prev_dir_x = 0
                # elif pos_xy[0] > 4.0:
                #     pos_xy[0] += -DISPLACEMENT
                #     prev_dir_x = 1
                # elif prev_dir_x == 0 and pos_xy[0] < 4.0:
                #     pos_xy[0] += DISPLACEMENT
                #     prev_dir_x = 0
                # elif prev_dir_x == 1 and pos_xy[0] > -4.0:
                #     pos_xy[0] += -DISPLACEMENT
                #     prev_dir_x = 1
                #
                # if pos_xy[1] < -4.0:
                #     pos_xy[1] += DISPLACEMENT
                #     prev_dir_y = 0
                # elif pos_xy[1] > 4.0:
                #     pos_xy[1] += -DISPLACEMENT
                #     prev_dir_y = 1
                # elif prev_dir_y == 0 and pos_xy[1] < 4.0:
                #     pos_xy[1] += DISPLACEMENT
                #     prev_dir_y = 0
                # elif prev_dir_y == 1 and pos_xy[1] > -4.0:
                #     pos_xy[1] += -DISPLACEMENT
                #     prev_dir_y = 1

                # This will adjust as needed per iteration
                loop_rate.sleep()

        except KeyboardInterrupt:
            pass


def euler_to_quaternion(roll, pitch, yaw):
    qx = sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2) - cos(roll / 2) * sin(
        pitch / 2
    ) * sin(yaw / 2)
    qy = cos(roll / 2) * sin(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * cos(
        pitch / 2
    ) * sin(yaw / 2)
    qz = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) - sin(roll / 2) * sin(
        pitch / 2
    ) * cos(yaw / 2)
    qw = cos(roll / 2) * cos(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * sin(
        pitch / 2
    ) * sin(yaw / 2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def main():
    node = StatePublisher()


if __name__ == "__main__":
    main()
