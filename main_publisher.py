import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

init_pos_table = [0.0, 0.0, 0.5]
init_pos_chair = [0.0, 1.0, 0.5]
init_pos_robot = [2.0, 0.0, 0.5]

pos_table = init_pos_table
pos_chair = init_pos_chair
pos_robot = init_pos_robot

rot_all = [0.0, 0.0, 0.0]

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        # super().__init__('fixed_frame_tf2_broadcaster')

        self.br = TransformBroadcaster(self)
        # self.timer = self.create_timer(0.1, self.broadcast_timer_callback)

        #self.init_pos('base_table', [0.0, 0.0, 0.5])
        #self.init_pos('base_chair', [0.5, 0.5, 0.5])
        #self.init_pos('base_robot', [0.5, 0.0, 0.5])
    
    def init_pos(self, link, pos):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = link
        print(pos[0])
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)

    def move_pos(self, link, trans, rot):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = link
        print(rot[0])
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = 1.0

    def timer_callback(self):
        t = TransformStamped()

        #self.move_pos('base_table', pos_table, rot_all)
        #self.move_pos('base_chair', pos_chair, rot_all)
        #self.move_pos('base_robot', pos_robot, rot_all)

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_chair'
        t.transform.translation.x = pos_chair[0]
        t.transform.translation.y = pos_chair[1]
        t.transform.translation.z = pos_chair[2]
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)


        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_robot'
        t.transform.translation.x = pos_robot[0]
        t.transform.translation.y = pos_robot[1]
        t.transform.translation.z = pos_robot[2]
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_table'
        t.transform.translation.x = pos_table[0]
        t.transform.translation.y = pos_table[1]
        t.transform.translation.z = pos_table[2]
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)
        
        #print("Doing another main.")

        pos_table[0] += 0.05
        pos_robot[0] += 0.05
        pos_chair[0] += 0.05


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()