#!/usr/bin/env python3

"""
Central Nervous Node for Humanoid Robot

This node serves as the primary coordination hub for the humanoid robot's
nervous system, managing communication between different subsystems and
ensuring proper timing and synchronization of operations.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import time


class CentralNervousNode(Node):
    def __init__(self):
        super().__init__('central_nervous_node')

        # Publishers for system status and coordination
        self.status_publisher = self.create_publisher(String, 'system_status', 10)
        self.emergency_stop_publisher = self.create_publisher(Bool, 'emergency_stop', 10)

        # Subscribers for monitoring system state
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.motion_command_subscriber = self.create_subscription(
            Twist, 'motion_commands', self.motion_command_callback, 10)

        # Timer for system monitoring
        self.system_monitor_timer = self.create_timer(0.1, self.system_monitor_callback)

        # System state tracking
        self.system_status = 'INITIALIZING'
        self.joint_states = None
        self.motion_commands = None
        self.last_update_time = time.time()

        self.get_logger().info('Central Nervous Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        self.joint_states = msg
        self.last_update_time = time.time()

        # Check for joint anomalies
        if self.detect_joint_anomaly(msg):
            self.handle_joint_anomaly(msg)

    def motion_command_callback(self, msg):
        """Callback for motion commands"""
        self.motion_commands = msg
        self.last_update_time = time.time()

    def system_monitor_callback(self):
        """Timer callback for system monitoring"""
        current_time = time.time()

        # Check if system is responsive
        if current_time - self.last_update_time > 1.0:  # 1 second timeout
            self.system_status = 'UNRESPONSIVE'
            self.emergency_stop_publisher.publish(Bool(data=True))
        else:
            self.system_status = 'OPERATIONAL'

        # Publish system status
        status_msg = String()
        status_msg.data = f"Status: {self.system_status}, Time: {current_time}"
        self.status_publisher.publish(status_msg)

    def detect_joint_anomaly(self, joint_state):
        """Detect potential joint anomalies"""
        # Simple anomaly detection - could be expanded
        for i, position in enumerate(joint_state.position):
            # Check for extreme values that might indicate sensor failure
            if abs(position) > 100:  # Arbitrary threshold
                self.get_logger().warn(f'Joint {i} position anomaly detected: {position}')
                return True
        return False

    def handle_joint_anomaly(self, joint_state):
        """Handle detected joint anomalies"""
        self.get_logger().error('Joint anomaly detected - initiating safety protocol')
        # Publish emergency stop
        self.emergency_stop_publisher.publish(Bool(data=True))
        self.system_status = 'SAFETY_LOCKOUT'


def main(args=None):
    rclpy.init(args=args)

    central_nervous_node = CentralNervousNode()

    try:
        rclpy.spin(central_nervous_node)
    except KeyboardInterrupt:
        pass
    finally:
        central_nervous_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()