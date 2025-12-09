#!/usr/bin/env python3
# safety_system.py
# Safety and validation system for humanoid robot

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool, Float64
from builtin_interfaces.msg import Duration
import numpy as np
import threading
import time
from typing import List, Dict, Tuple

class SafetyMonitor(Node):
    """
    Safety monitoring system for humanoid robot
    """
    def __init__(self):
        super().__init__('safety_monitor')

        # Parameters
        self.declare_parameter('emergency_stop_timeout', 0.1)  # seconds
        self.declare_parameter('max_joint_velocity', 5.0)  # rad/s
        self.declare_parameter('max_joint_effort', 100.0)  # Nm
        self.declare_parameter('max_angular_velocity', 3.0)  # rad/s (IMU)
        self.declare_parameter('max_linear_acceleration', 20.0)  # m/s^2 (IMU)
        self.declare_parameter('collision_force_threshold', 50.0)  # N
        self.declare_parameter('fall_angle_threshold', 0.5)  # rad (~30 degrees)

        self.emergency_stop_timeout = self.get_parameter('emergency_stop_timeout').value
        self.max_joint_velocity = self.get_parameter('max_joint_velocity').value
        self.max_joint_effort = self.get_parameter('max_joint_effort').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.max_linear_acceleration = self.get_parameter('max_linear_acceleration').value
        self.collision_force_threshold = self.get_parameter('collision_force_threshold').value
        self.fall_angle_threshold = self.get_parameter('fall_angle_threshold').value

        # Joint state monitoring
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # IMU monitoring
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Force/torque monitoring
        self.ft_sub = self.create_subscription(
            WrenchStamped,
            '/left_foot/ft_sensor',
            self.ft_callback,
            10
        )

        # Additional FT sensor for right foot
        self.right_ft_sub = self.create_subscription(
            WrenchStamped,
            '/right_foot/ft_sensor',
            self.ft_callback,
            10
        )

        # Emergency stop publisher
        qos_profile = QoSProfile(depth=1)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', qos_profile)

        # Joint limits publisher
        self.joint_limits_pub = self.create_publisher(JointState, '/joint_limits_violation', 10)

        # Initialize state variables
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_joint_efforts = {}
        self.last_joint_state_time = self.get_clock().now()
        self.emergency_stop_active = False
        self.fall_detected = False
        self.collision_detected = False

        # Create timer for safety checks
        self.safety_timer = self.create_timer(0.01, self.safety_check)  # 100Hz

        self.get_logger().info('Safety Monitor initialized')

    def joint_state_callback(self, msg: JointState):
        """
        Callback for joint state messages
        """
        self.last_joint_state_time = self.get_clock().now()

        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_joint_efforts[name] = msg.effort[i]

    def imu_callback(self, msg: Imu):
        """
        Callback for IMU messages
        """
        # Check angular velocity limits
        angular_vel = np.sqrt(
            msg.angular_velocity.x**2 +
            msg.angular_velocity.y**2 +
            msg.angular_velocity.z**2
        )

        if angular_vel > self.max_angular_velocity:
            self.get_logger().warn(f'Angular velocity limit exceeded: {angular_vel} > {self.max_angular_velocity}')
            self.trigger_emergency_stop()

        # Check linear acceleration limits
        linear_acc = np.sqrt(
            msg.linear_acceleration.x**2 +
            msg.linear_acceleration.y**2 +
            msg.linear_acceleration.z**2
        )

        if linear_acc > self.max_linear_acceleration:
            self.get_logger().warn(f'Linear acceleration limit exceeded: {linear_acc} > {self.max_linear_acceleration}')
            self.trigger_emergency_stop()

        # Check for potential fall (simplified: check roll/pitch angles)
        # Convert quaternion to roll/pitch
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        roll, pitch, _ = self.quaternion_to_euler(quat)

        if abs(roll) > self.fall_angle_threshold or abs(pitch) > self.fall_angle_threshold:
            self.get_logger().warn(f'Potential fall detected: roll={roll}, pitch={pitch}')
            self.fall_detected = True

    def ft_callback(self, msg: WrenchStamped):
        """
        Callback for force/torque sensor messages
        """
        force_magnitude = np.sqrt(
            msg.wrench.force.x**2 +
            msg.wrench.force.y**2 +
            msg.wrench.force.z**2
        )

        if force_magnitude > self.collision_force_threshold:
            self.get_logger().warn(f'Collision detected: force={force_magnitude} > {self.collision_force_threshold}')
            self.collision_detected = True
            self.trigger_emergency_stop()

    def safety_check(self):
        """
        Perform safety checks at regular intervals
        """
        if self.emergency_stop_active:
            return

        # Check for joint limit violations
        for joint_name, velocity in self.current_joint_velocities.items():
            if abs(velocity) > self.max_joint_velocity:
                self.get_logger().warn(f'Joint velocity limit exceeded for {joint_name}: {velocity} > {self.max_joint_velocity}')

        for joint_name, effort in self.current_joint_efforts.items():
            if abs(effort) > self.max_joint_effort:
                self.get_logger().warn(f'Joint effort limit exceeded for {joint_name}: {effort} > {self.max_joint_effort}')

        # Check for communication timeout
        time_since_last_joint_state = (
            self.get_clock().now() - self.last_joint_state_time
        ).nanoseconds / 1e9

        if time_since_last_joint_state > self.emergency_stop_timeout:
            self.get_logger().error('Joint state timeout - emergency stop triggered')
            self.trigger_emergency_stop()

        # Check for fall detection
        if self.fall_detected:
            self.get_logger().warn('Fall detected - emergency stop triggered')
            self.trigger_emergency_stop()

        # Check for collision detection
        if self.collision_detected:
            self.get_logger().warn('Collision detected - emergency stop triggered')
            self.trigger_emergency_stop()

    def trigger_emergency_stop(self):
        """
        Trigger emergency stop
        """
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            msg = Bool()
            msg.data = True
            self.emergency_stop_pub.publish(msg)
            self.get_logger().error('EMERGENCY STOP ACTIVATED')

    def reset_emergency_stop(self):
        """
        Reset emergency stop (only after addressing the issue)
        """
        self.emergency_stop_active = False
        self.fall_detected = False
        self.collision_detected = False
        msg = Bool()
        msg.data = False
        self.emergency_stop_pub.publish(msg)
        self.get_logger().info('Emergency stop reset')

    def quaternion_to_euler(self, quat: List[float]) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        """
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


class ValidationSystem(Node):
    """
    Validation system for humanoid robot operations
    """
    def __init__(self):
        super().__init__('validation_system')

        # Parameters
        self.declare_parameter('validation_period', 1.0)  # seconds
        self.validation_period = self.get_parameter('validation_period').value

        # Publishers and subscribers would be defined here
        # For this example, we'll just create a timer for validation checks

        self.validation_timer = self.create_timer(
            self.validation_period,
            self.validation_check
        )

        self.get_logger().info('Validation System initialized')

    def validation_check(self):
        """
        Perform validation checks
        """
        self.get_logger().info('Performing validation check...')

        # Placeholder for validation logic
        # In a real system, this would check:
        # - Sensor data validity
        # - Actuator health
        # - Communication integrity
        # - Performance metrics
        # - Safety system status

        # Example validation: Check if safety monitor is responsive
        # (This would involve checking if the safety monitor is publishing status)

        self.get_logger().info('Validation check completed')


def main(args=None):
    rclpy.init(args=args)

    safety_monitor = SafetyMonitor()
    validation_system = ValidationSystem()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(safety_monitor)
    executor.add_node(validation_system)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        safety_monitor.destroy_node()
        validation_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()