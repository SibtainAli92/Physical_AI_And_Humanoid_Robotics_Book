#!/usr/bin/env python3

"""
Sensor Fusion Node for Humanoid Robot

This node integrates data from multiple sensors to create a coherent
understanding of the robot's state and environment, similar to how
the human brain combines sensory inputs.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from geometry_msgs.msg import Vector3, PointStamped
from std_msgs.msg import Float32
import numpy as np
from collections import deque
import math


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Publishers for fused sensor data
        self.fused_orientation_publisher = self.create_publisher(Vector3, 'fused_orientation', 10)
        self.center_of_mass_publisher = self.create_publisher(PointStamped, 'center_of_mass', 10)
        self.balance_state_publisher = self.create_publisher(Float32, 'balance_state', 10)

        # Subscribers for raw sensor data
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.laser_scan_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_scan_callback, 10)

        # Timer for fusion calculations
        self.fusion_timer = self.create_timer(0.05, self.fusion_callback)  # 20 Hz

        # Sensor data storage
        self.imu_data = None
        self.joint_states = None
        self.laser_scan = None

        # Joint position history for stability analysis
        self.joint_history = deque(maxlen=10)

        # Robot model parameters (simplified)
        self.robot_mass = 50.0  # kg
        self.joint_positions = {}  # joint name to position mapping
        self.joint_masses = {}    # joint name to mass mapping

        self.get_logger().info('Sensor Fusion Node initialized')

    def imu_callback(self, msg):
        """Callback for IMU data"""
        self.imu_data = msg

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        self.joint_states = msg
        # Store current joint positions
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

        # Store joint positions for history
        self.joint_history.append(dict(self.joint_positions))

    def laser_scan_callback(self, msg):
        """Callback for laser scan data"""
        self.laser_scan = msg

    def fusion_callback(self):
        """Timer callback for sensor fusion calculations"""
        if self.imu_data is not None and self.joint_states is not None:
            # Calculate fused orientation from IMU and joint data
            fused_orientation = self.calculate_fused_orientation()
            self.fused_orientation_publisher.publish(fused_orientation)

            # Calculate center of mass based on joint positions
            com = self.calculate_center_of_mass()
            self.center_of_mass_publisher.publish(com)

            # Calculate balance state
            balance_state = self.calculate_balance_state()
            self.balance_state_publisher.publish(balance_state)

    def calculate_fused_orientation(self):
        """Calculate orientation by fusing IMU and joint data"""
        # Extract orientation from IMU (quaternion to Euler angles)
        imu = self.imu_data
        w, x, y, z = imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z

        # Convert quaternion to Euler angles
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Create Vector3 message with fused orientation
        orientation = Vector3()
        orientation.x = roll
        orientation.y = pitch
        orientation.z = yaw

        return orientation

    def calculate_center_of_mass(self):
        """Calculate center of mass based on joint positions and masses"""
        # Simplified calculation - in reality, this would use a more complex kinematic model
        total_mass = self.robot_mass
        com_x = 0.0
        com_y = 0.0
        com_z = 0.0

        # Add contribution from each joint based on position and estimated mass
        for joint_name, position in self.joint_positions.items():
            # Simplified mass distribution - in reality each joint would have different mass
            joint_mass = 1.0  # Simplified
            total_mass += joint_mass

            # Simplified position calculation - would use forward kinematics in reality
            if 'hip' in joint_name:
                com_z += position * joint_mass
            elif 'shoulder' in joint_name:
                com_x += position * joint_mass
            elif 'knee' in joint_name:
                com_z += position * joint_mass

        # Normalize by total mass
        com_x /= total_mass
        com_y /= total_mass
        com_z /= total_mass

        # Create PointStamped message
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'base_link'
        point.point.x = com_x
        point.point.y = com_y
        point.point.z = com_z

        return point

    def calculate_balance_state(self):
        """Calculate balance state based on orientation and center of mass"""
        if self.imu_data is None or not self.joint_positions:
            balance_state = Float32()
            balance_state.data = 0.0  # Neutral balance
            return balance_state

        # Calculate balance based on tilt angles from IMU
        orientation = self.calculate_fused_orientation()
        tilt_angle = math.sqrt(orientation.x**2 + orientation.y**2)  # Combined roll and pitch

        # Calculate zero-moment point (simplified)
        com = self.calculate_center_of_mass()

        # Balance score (0.0 = perfectly balanced, 1.0 = unstable)
        balance_score = min(1.0, tilt_angle / (math.pi/4))  # Normalize to 45-degree max tilt

        # Adjust based on center of mass position
        if abs(com.point.x) > 0.1 or abs(com.point.z) > 0.1:  # 10cm threshold
            balance_score = min(1.0, balance_score + 0.2)

        balance_state = Float32()
        balance_state.data = balance_score

        return balance_state


def main(args=None):
    rclpy.init(args=args)

    sensor_fusion_node = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()