#!/usr/bin/env python3

"""
Behavior Manager Node for Humanoid Robot

This node manages high-level behaviors and state transitions for the humanoid robot,
similar to how the brain manages different behavioral states and transitions.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
from enum import Enum
import threading


class RobotBehavior(Enum):
    IDLE = "idle"
    WALKING = "walking"
    STANDING = "standing"
    SITTING = "sitting"
    BALANCING = "balancing"
    EMERGENCY = "emergency"


class BehaviorManagerNode(Node):
    def __init__(self):
        super().__init__('behavior_manager_node')

        # Publishers for behavior commands
        self.behavior_state_publisher = self.create_publisher(String, 'behavior_state', 10)
        self.motion_command_publisher = self.create_publisher(Twist, 'motion_commands', 10)
        self.odometry_publisher = self.create_publisher(Odometry, 'odom', 10)

        # Subscribers for sensor and status feedback
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.balance_state_subscriber = self.create_subscription(
            Float32, 'balance_state', self.balance_state_callback, 10)
        self.emergency_stop_subscriber = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10)

        # Timer for behavior management
        self.behavior_timer = self.create_timer(0.1, self.behavior_management_callback)  # 10 Hz

        # Behavior state tracking
        self.current_behavior = RobotBehavior.IDLE
        self.previous_behavior = RobotBehavior.IDLE
        self.behavior_start_time = time.time()
        self.joint_states = None
        self.balance_state = 0.0
        self.emergency_stop_active = False
        self.robot_position = [0.0, 0.0, 0.0]  # x, y, theta
        self.robot_velocity = [0.0, 0.0, 0.0]  # linear_x, linear_y, angular_z

        # Behavior parameters
        self.balance_threshold = 0.7  # Above this value, robot is unstable
        self.transition_lock = threading.Lock()

        self.get_logger().info('Behavior Manager Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        self.joint_states = msg

    def balance_state_callback(self, msg):
        """Callback for balance state updates"""
        self.balance_state = msg.data

    def emergency_stop_callback(self, msg):
        """Callback for emergency stop commands"""
        self.emergency_stop_active = msg.data
        if self.emergency_stop_active:
            self.transition_to_behavior(RobotBehavior.EMERGENCY)

    def behavior_management_callback(self):
        """Timer callback for behavior management"""
        if self.emergency_stop_active:
            return

        # Evaluate current state and determine if behavior transition is needed
        new_behavior = self.evaluate_behavior_transition()

        # Transition to new behavior if needed
        if new_behavior != self.current_behavior:
            self.transition_to_behavior(new_behavior)

        # Execute current behavior
        self.execute_current_behavior()

        # Publish current behavior state
        behavior_msg = String()
        behavior_msg.data = f"{self.current_behavior.value}:{time.time() - self.behavior_start_time:.2f}s"
        self.behavior_state_publisher.publish(behavior_msg)

        # Publish odometry (simplified)
        self.publish_odometry()

    def evaluate_behavior_transition(self):
        """Evaluate current state and determine appropriate behavior"""
        # Check for balance issues
        if self.balance_state > self.balance_threshold:
            return RobotBehavior.BALANCING

        # Check for standing position
        if self.current_behavior == RobotBehavior.STANDING:
            # Check if joints indicate we're still in standing position
            if self.joint_states:
                # Simplified check - in reality would check specific joint angles
                standing_position = True  # Placeholder logic
                if not standing_position:
                    return RobotBehavior.IDLE

        # Default behavior based on current state
        if self.current_behavior == RobotBehavior.EMERGENCY:
            return RobotBehavior.IDLE  # Return to idle after emergency

        # Default to current behavior if no transition needed
        return self.current_behavior

    def transition_to_behavior(self, new_behavior):
        """Handle transition to a new behavior"""
        with self.transition_lock:
            self.get_logger().info(f'Transitioning from {self.current_behavior.value} to {new_behavior.value}')

            # Execute exit procedures for current behavior
            self.exit_current_behavior()

            # Store previous behavior
            self.previous_behavior = self.current_behavior
            self.current_behavior = new_behavior
            self.behavior_start_time = time.time()

            # Execute entry procedures for new behavior
            self.enter_new_behavior()

    def exit_current_behavior(self):
        """Execute exit procedures for current behavior"""
        # Stop any ongoing motion
        stop_cmd = Twist()
        self.motion_command_publisher.publish(stop_cmd)

    def enter_new_behavior(self):
        """Execute entry procedures for new behavior"""
        if self.current_behavior == RobotBehavior.STANDING:
            self.assume_standing_pose()
        elif self.current_behavior == RobotBehavior.BALANCING:
            self.activate_balance_control()
        elif self.current_behavior == RobotBehavior.WALKING:
            self.prepare_for_locomotion()

    def execute_current_behavior(self):
        """Execute the current behavior"""
        if self.current_behavior == RobotBehavior.IDLE:
            self.execute_idle_behavior()
        elif self.current_behavior == RobotBehavior.WALKING:
            self.execute_walking_behavior()
        elif self.current_behavior == RobotBehavior.STANDING:
            self.execute_standing_behavior()
        elif self.current_behavior == RobotBehavior.BALANCING:
            self.execute_balancing_behavior()
        elif self.current_behavior == RobotBehavior.EMERGENCY:
            self.execute_emergency_behavior()

    def execute_idle_behavior(self):
        """Execute idle behavior"""
        # In idle state, maintain a neutral position
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.motion_command_publisher.publish(cmd)

    def execute_walking_behavior(self):
        """Execute walking behavior"""
        # Simplified walking command - would be more complex in reality
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd.angular.z = 0.0
        self.motion_command_publisher.publish(cmd)

    def execute_standing_behavior(self):
        """Execute standing behavior"""
        # Maintain standing position
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.motion_command_publisher.publish(cmd)

    def execute_balancing_behavior(self):
        """Execute balancing behavior"""
        # Simplified balance control - in reality would use more sophisticated algorithms
        cmd = Twist()

        # Counteract tilt based on balance state
        if self.balance_state > 0.5:
            # Apply corrective motion based on balance state
            cmd.angular.z = -0.2 * self.balance_state  # Counteract rotation
            cmd.linear.x = -0.1 * self.balance_state   # Counteract forward/back tilt
        else:
            cmd.angular.z = 0.0
            cmd.linear.x = 0.0

        self.motion_command_publisher.publish(cmd)

    def execute_emergency_behavior(self):
        """Execute emergency behavior"""
        # Complete stop and safe position
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.motion_command_publisher.publish(cmd)

    def assume_standing_pose(self):
        """Command robot to assume standing pose"""
        self.get_logger().info('Assuming standing pose')

    def activate_balance_control(self):
        """Activate balance control systems"""
        self.get_logger().info('Activating balance control')

    def prepare_for_locomotion(self):
        """Prepare systems for locomotion"""
        self.get_logger().info('Preparing for locomotion')

    def publish_odometry(self):
        """Publish odometry information"""
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # Set position (simplified)
        odom.pose.pose.position.x = self.robot_position[0]
        odom.pose.pose.position.y = self.robot_position[1]
        odom.pose.pose.position.z = 0.0  # Assuming flat ground

        # Set orientation (simplified - using Euler angles converted to quaternion)
        # For now, just set to neutral
        odom.pose.pose.orientation.w = 1.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = 0.0

        # Set velocity
        odom.twist.twist.linear.x = self.robot_velocity[0]
        odom.twist.twist.linear.y = self.robot_velocity[1]
        odom.twist.twist.angular.z = self.robot_velocity[2]

        self.odometry_publisher.publish(odom)

    def request_behavior_change(self, new_behavior):
        """Request a change to a specific behavior"""
        if not self.emergency_stop_active:
            self.transition_to_behavior(new_behavior)


def main(args=None):
    rclpy.init(args=args)

    behavior_manager_node = BehaviorManagerNode()

    try:
        rclpy.spin(behavior_manager_node)
    except KeyboardInterrupt:
        pass
    finally:
        behavior_manager_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()