#!/usr/bin/env python3

"""
Motor Control Node for Humanoid Robot

This node handles the control of actuators and motors, providing
low-level control interfaces and safety mechanisms for the humanoid robot.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import time
import threading


class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control_node')

        # Publishers for motor control
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.trajectory_publisher = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.controller_state_publisher = self.create_publisher(JointTrajectoryControllerState, 'controller_state', 10)

        # Subscribers for control commands
        self.joint_trajectory_subscriber = self.create_subscription(
            JointTrajectory, 'desired_joint_trajectory', self.joint_trajectory_callback, 10)
        self.emergency_stop_subscriber = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10)

        # Timer for motor control loop
        self.control_timer = self.create_timer(0.01, self.control_callback)  # 100 Hz

        # Motor state tracking
        self.current_positions = {}
        self.current_velocities = {}
        self.current_efforts = {}
        self.desired_trajectory = None
        self.trajectory_index = 0
        self.emergency_stop_active = False
        self.last_command_time = time.time()

        # Joint limits and safety parameters
        self.joint_limits = {
            'hip_joint': (-1.57, 1.57),      # ±90 degrees
            'knee_joint': (0, 2.35),         # 0 to 135 degrees
            'ankle_joint': (-0.78, 0.78),    # ±45 degrees
            'shoulder_joint': (-1.57, 1.57), # ±90 degrees
            'elbow_joint': (0, 2.35),        # 0 to 135 degrees
        }

        # Initialize with default positions
        for joint_name in self.joint_limits.keys():
            self.current_positions[joint_name] = 0.0
            self.current_velocities[joint_name] = 0.0
            self.current_efforts[joint_name] = 0.0

        self.get_logger().info('Motor Control Node initialized')

    def joint_trajectory_callback(self, msg):
        """Callback for joint trajectory commands"""
        if not self.emergency_stop_active:
            self.desired_trajectory = msg
            self.trajectory_index = 0
            self.last_command_time = time.time()
            self.get_logger().info(f'Received trajectory with {len(msg.points)} points')

    def emergency_stop_callback(self, msg):
        """Callback for emergency stop commands"""
        self.emergency_stop_active = msg.data
        if self.emergency_stop_active:
            self.get_logger().warn('Emergency stop activated - stopping all motors')
            self.stop_all_motors()
        else:
            self.get_logger().info('Emergency stop released')

    def control_callback(self):
        """Timer callback for motor control"""
        if self.emergency_stop_active:
            return

        # Update current state based on trajectory or hold position
        self.update_motor_positions()

        # Publish current joint states
        self.publish_joint_states()

        # Publish controller state
        self.publish_controller_state()

        # Check for timeout
        if time.time() - self.last_command_time > 5.0:  # 5 second timeout
            self.stop_all_motors()

    def update_motor_positions(self):
        """Update motor positions based on trajectory or hold current position"""
        if self.desired_trajectory and self.trajectory_index < len(self.desired_trajectory.points):
            # Execute trajectory point
            point = self.desired_trajectory.points[self.trajectory_index]

            # Check if we've reached the time for this trajectory point
            current_time = time.time()
            trajectory_start = self.desired_trajectory.header.stamp.sec + self.desired_trajectory.header.stamp.nanosec / 1e9
            point_time = trajectory_start + point.time_from_start.sec + point.time_from_start.nanosec / 1e9

            if current_time >= point_time:
                # Update positions to trajectory point values
                for i, joint_name in enumerate(self.desired_trajectory.joint_names):
                    if i < len(point.positions):
                        desired_pos = point.positions[i]
                        # Apply joint limits
                        min_limit, max_limit = self.joint_limits.get(joint_name, (-100, 100))
                        desired_pos = max(min_limit, min(max_limit, desired_pos))

                        # Update position with simple interpolation
                        current_pos = self.current_positions.get(joint_name, desired_pos)
                        new_pos = self.interpolate_position(current_pos, desired_pos, 0.1)  # 10% interpolation

                        self.current_positions[joint_name] = new_pos
                        self.current_velocities[joint_name] = 0.0  # Simplified velocity calculation
                        self.current_efforts[joint_name] = 0.0    # Simplified effort calculation

                self.trajectory_index += 1
        else:
            # Hold current positions - apply gentle control to maintain position
            for joint_name in self.current_positions.keys():
                # Apply position holding control (simplified)
                current_pos = self.current_positions[joint_name]
                self.current_positions[joint_name] = current_pos
                self.current_velocities[joint_name] = 0.0
                self.current_efforts[joint_name] = 0.0

    def interpolate_position(self, current, target, factor):
        """Simple interpolation between current and target position"""
        return current + factor * (target - current)

    def publish_joint_states(self):
        """Publish current joint states"""
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_link'

        for joint_name in sorted(self.current_positions.keys()):
            joint_state.name.append(joint_name)
            joint_state.position.append(self.current_positions[joint_name])
            joint_state.velocity.append(self.current_velocities[joint_name])
            joint_state.effort.append(self.current_efforts[joint_name])

        self.joint_command_publisher.publish(joint_state)

    def publish_controller_state(self):
        """Publish controller state"""
        controller_state = JointTrajectoryControllerState()
        controller_state.header.stamp = self.get_clock().now().to_msg()
        controller_state.header.frame_id = 'base_link'

        for joint_name in sorted(self.current_positions.keys()):
            controller_state.joint_names.append(joint_name)
            controller_state.desired.positions.append(self.current_positions[joint_name])
            controller_state.desired.velocities.append(self.current_velocities[joint_name])
            controller_state.actual.positions.append(self.current_positions[joint_name])
            controller_state.actual.velocities.append(self.current_velocities[joint_name])
            controller_state.error.positions.append(0.0)  # Simplified error calculation
            controller_state.error.velocities.append(0.0)

        self.controller_state_publisher.publish(controller_state)

    def stop_all_motors(self):
        """Stop all motors immediately"""
        for joint_name in self.current_positions.keys():
            self.current_positions[joint_name] = self.current_positions[joint_name]  # Hold position
            self.current_velocities[joint_name] = 0.0
            self.current_efforts[joint_name] = 0.0

        # Publish stop command
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_link'

        for joint_name in sorted(self.current_positions.keys()):
            joint_state.name.append(joint_name)
            joint_state.position.append(self.current_positions[joint_name])
            joint_state.velocity.append(0.0)  # Stop velocity
            joint_state.effort.append(0.0)    # Zero effort

        self.joint_command_publisher.publish(joint_state)

    def check_joint_limits(self, joint_name, position):
        """Check if joint position is within limits"""
        if joint_name in self.joint_limits:
            min_limit, max_limit = self.joint_limits[joint_name]
            return min_limit <= position <= max_limit
        return True  # If no limit defined, assume valid


def main(args=None):
    rclpy.init(args=args)

    motor_control_node = MotorControlNode()

    try:
        rclpy.spin(motor_control_node)
    except KeyboardInterrupt:
        pass
    finally:
        motor_control_node.stop_all_motors()
        motor_control_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()