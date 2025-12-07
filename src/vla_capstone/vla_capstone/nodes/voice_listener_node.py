import rclpy
from rclpy.node import Node


class VoiceListenerNode(Node):

    def __init__(self):
        super().__init__('voice_listener_node')
        self.get_logger().info('Voice Listener Node Started')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()