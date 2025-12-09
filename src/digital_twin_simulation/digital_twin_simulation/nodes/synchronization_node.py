#!/usr/bin/env python3

"""
Synchronization Node for Humanoid Robot Digital Twin

This node handles synchronization between the physical robot and its digital twin,
managing data flow, time synchronization, and state consistency between real and
virtual systems.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Header
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Vector3, Point
from nav_msgs.msg import Odometry
import time
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import threading


@dataclass
class PhysicalState:
    """Represents the state of the physical robot"""
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]
    joint_efforts: Dict[str, float]
    position: Point
    orientation: Vector3
    linear_velocity: Vector3
    angular_velocity: Vector3
    timestamp: float
    sequence: int


@dataclass
class VirtualState:
    """Represents the state of the virtual robot"""
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]
    joint_efforts: Dict[str, float]
    position: Point
    orientation: Vector3
    linear_velocity: Vector3
    angular_velocity: Vector3
    timestamp: float
    sequence: int


class SynchronizationNode(Node):
    def __init__(self):
        super().__init__('synchronization_node')

        # Publishers for synchronization data
        self.sync_status_publisher = self.create_publisher(String, 'sync/status', 10)
        self.physical_to_virtual_publisher = self.create_publisher(JointState, 'sync/physical_to_virtual', 10)
        self.virtual_to_physical_publisher = self.create_publisher(JointState, 'sync/virtual_to_physical', 10)
        self.time_offset_publisher = self.create_publisher(Float32, 'sync/time_offset', 10)

        # Subscribers for physical robot data
        self.physical_joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.physical_joint_state_callback, 10)
        self.physical_imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.physical_imu_callback, 10)
        self.physical_odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.physical_odom_callback, 10)

        # Subscribers for virtual robot data
        self.virtual_joint_state_subscriber = self.create_subscription(
            JointState, 'sim/joint_states', self.virtual_joint_state_callback, 10)
        self.virtual_imu_subscriber = self.create_subscription(
            Imu, 'sim/imu/data', self.virtual_imu_callback, 10)
        self.virtual_odom_subscriber = self.create_subscription(
            Odometry, 'sim/odom', self.virtual_odom_callback, 10)

        # Timer for synchronization management
        self.sync_timer = self.create_timer(0.05, self.synchronization_callback)  # 20 Hz

        # Synchronization state
        self.physical_state: Optional[PhysicalState] = None
        self.virtual_state: Optional[VirtualState] = None
        self.last_sync_time = time.time()
        self.time_offset = 0.0  # Offset between physical and virtual time
        self.sequence_counter = 0
        self.synchronization_enabled = True
        self.sync_mode = 'mirror'  # 'mirror', 'predict', 'compare'
        self.sync_threshold = 0.1  # Threshold for synchronization correction

        # State history for analysis
        self.physical_history = []
        self.virtual_history = []
        self.max_history_length = 100

        # Statistics
        self.sync_errors = []
        self.avg_sync_error = 0.0
        self.max_sync_error = 0.0

        # Threading lock for state access
        self.state_lock = threading.Lock()

        self.get_logger().info('Synchronization Node initialized')

    def physical_joint_state_callback(self, msg):
        """Callback for physical robot joint state updates"""
        with self.state_lock:
            if self.physical_state is None:
                self.physical_state = PhysicalState(
                    joint_positions={},
                    joint_velocities={},
                    joint_efforts={},
                    position=Point(x=0.0, y=0.0, z=0.0),
                    orientation=Vector3(x=0.0, y=0.0, z=0.0),
                    linear_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                    angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                    timestamp=time.time(),
                    sequence=self.sequence_counter
                )

            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.physical_state.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.physical_state.joint_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.physical_state.joint_efforts[name] = msg.effort[i]

            self.physical_state.timestamp = time.time()
            self.physical_state.sequence = self.sequence_counter
            self.sequence_counter += 1

            # Add to history
            self.physical_history.append((self.physical_state.timestamp, dict(self.physical_state.joint_positions)))
            if len(self.physical_history) > self.max_history_length:
                self.physical_history.pop(0)

    def physical_imu_callback(self, msg):
        """Callback for physical robot IMU updates"""
        with self.state_lock:
            if self.physical_state:
                self.physical_state.linear_velocity.x = msg.linear_acceleration.x / 10  # Rough integration
                self.physical_state.linear_velocity.y = msg.linear_acceleration.y / 10
                self.physical_state.linear_velocity.z = msg.linear_acceleration.z / 10
                self.physical_state.angular_velocity = msg.angular_velocity

    def physical_odom_callback(self, msg):
        """Callback for physical robot odometry updates"""
        with self.state_lock:
            if self.physical_state:
                self.physical_state.position = msg.pose.pose.position
                # Convert quaternion to euler for orientation vector (simplified)
                # In reality, would maintain quaternion
                self.physical_state.orientation.x = msg.pose.pose.orientation.x
                self.physical_state.orientation.y = msg.pose.pose.orientation.y
                self.physical_state.orientation.z = msg.pose.pose.orientation.z
                self.physical_state.linear_velocity = msg.twist.twist.linear
                self.physical_state.angular_velocity = msg.twist.twist.angular

    def virtual_joint_state_callback(self, msg):
        """Callback for virtual robot joint state updates"""
        with self.state_lock:
            if self.virtual_state is None:
                self.virtual_state = VirtualState(
                    joint_positions={},
                    joint_velocities={},
                    joint_efforts={},
                    position=Point(x=0.0, y=0.0, z=0.0),
                    orientation=Vector3(x=0.0, y=0.0, z=0.0),
                    linear_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                    angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                    timestamp=time.time(),
                    sequence=self.sequence_counter
                )

            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.virtual_state.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.virtual_state.joint_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.virtual_state.joint_efforts[name] = msg.effort[i]

            self.virtual_state.timestamp = time.time()
            self.virtual_state.sequence = self.sequence_counter
            self.sequence_counter += 1

            # Add to history
            self.virtual_history.append((self.virtual_state.timestamp, dict(self.virtual_state.joint_positions)))
            if len(self.virtual_history) > self.max_history_length:
                self.virtual_history.pop(0)

    def virtual_imu_callback(self, msg):
        """Callback for virtual robot IMU updates"""
        with self.state_lock:
            if self.virtual_state:
                self.virtual_state.linear_velocity.x = msg.linear_acceleration.x / 10
                self.virtual_state.linear_velocity.y = msg.linear_acceleration.y / 10
                self.virtual_state.linear_velocity.z = msg.linear_acceleration.z / 10
                self.virtual_state.angular_velocity = msg.angular_velocity

    def virtual_odom_callback(self, msg):
        """Callback for virtual robot odometry updates"""
        with self.state_lock:
            if self.virtual_state:
                self.virtual_state.position = msg.pose.pose.position
                self.virtual_state.orientation.x = msg.pose.pose.orientation.x
                self.virtual_state.orientation.y = msg.pose.pose.orientation.y
                self.virtual_state.orientation.z = msg.pose.pose.orientation.z
                self.virtual_state.linear_velocity = msg.twist.twist.linear
                self.virtual_state.angular_velocity = msg.twist.twist.angular

    def synchronization_callback(self):
        """Main synchronization callback"""
        if not self.synchronization_enabled:
            return

        current_time = time.time()
        sync_status = []

        with self.state_lock:
            # Calculate time offset
            if self.physical_state and self.virtual_state:
                self.time_offset = self.virtual_state.timestamp - self.physical_state.timestamp

                # Calculate synchronization error
                sync_error = self.calculate_sync_error()
                self.sync_errors.append(sync_error)
                if len(self.sync_errors) > 50:  # Keep last 50 errors for statistics
                    self.sync_errors.pop(0)

                # Calculate statistics
                if self.sync_errors:
                    self.avg_sync_error = sum(self.sync_errors) / len(self.sync_errors)
                    self.max_sync_error = max(self.sync_errors)

                # Apply synchronization based on mode
                if self.sync_mode == 'mirror':
                    self.apply_mirror_sync()
                elif self.sync_mode == 'predict':
                    self.apply_predictive_sync()
                elif self.sync_mode == 'compare':
                    self.apply_compare_sync()

                # Publish time offset
                offset_msg = Float32()
                offset_msg.data = self.time_offset
                self.time_offset_publisher.publish(offset_msg)

                sync_status.append(f"Offset: {self.time_offset:.3f}s")
                sync_status.append(f"Error: {sync_error:.3f}")
                sync_status.append(f"Mode: {self.sync_mode}")

        # Publish synchronization status
        status_msg = String()
        status_msg.data = f"SYNC: {', '.join(sync_status)}"
        self.sync_status_publisher.publish(status_msg)

        self.last_sync_time = current_time

    def calculate_sync_error(self) -> float:
        """Calculate the synchronization error between physical and virtual states"""
        if not self.physical_state or not self.virtual_state:
            return 0.0

        # Calculate error as the average difference in joint positions
        total_error = 0.0
        common_joints = set(self.physical_state.joint_positions.keys()) & set(self.virtual_state.joint_positions.keys())
        error_count = 0

        for joint in common_joints:
            phys_pos = self.physical_state.joint_positions[joint]
            virt_pos = self.virtual_state.joint_positions[joint]
            total_error += abs(phys_pos - virt_pos)
            error_count += 1

        return total_error / error_count if error_count > 0 else 0.0

    def apply_mirror_sync(self):
        """Apply mirror synchronization (virtual follows physical)"""
        if not self.physical_state or not self.virtual_state:
            return

        # Copy physical state to virtual state with some delay simulation
        sync_delay = 0.05  # 50ms delay simulation
        target_time = time.time() - sync_delay

        # Apply position corrections to virtual state to match physical
        for joint, phys_pos in self.physical_state.joint_positions.items():
            if joint in self.virtual_state.joint_positions:
                # Calculate correction factor based on synchronization error
                current_error = abs(phys_pos - self.virtual_state.joint_positions[joint])
                correction_factor = min(1.0, current_error / self.sync_threshold) if self.sync_threshold > 0 else 1.0

                # Apply correction
                self.virtual_state.joint_positions[joint] = (
                    self.virtual_state.joint_positions[joint] * (1 - correction_factor) +
                    phys_pos * correction_factor
                )

        # Update virtual state timestamp to reflect synchronization
        self.virtual_state.timestamp = time.time()

    def apply_predictive_sync(self):
        """Apply predictive synchronization (virtual predicts physical)"""
        if not self.physical_state or not self.virtual_state:
            return

        # Predict physical state based on virtual state and update physical accordingly
        # This would use more sophisticated prediction algorithms in a real system
        prediction_horizon = 0.1  # 100ms prediction horizon

        # Simple prediction based on velocity
        for joint, virt_pos in self.virtual_state.joint_positions.items():
            if joint in self.virtual_state.joint_velocities:
                predicted_pos = virt_pos + self.virtual_state.joint_velocities[joint] * prediction_horizon
                # Apply to physical state (in a real system, this would be sent to physical robot)
                pass

    def apply_compare_sync(self):
        """Apply compare synchronization (monitor only)"""
        # In compare mode, we just monitor the differences without correcting
        pass

    def set_sync_mode(self, mode: str):
        """Set the synchronization mode"""
        if mode in ['mirror', 'predict', 'compare']:
            old_mode = self.sync_mode
            self.sync_mode = mode
            self.get_logger().info(f'Synchronization mode changed from {old_mode} to {mode}')
        else:
            self.get_logger().warn(f'Invalid synchronization mode: {mode}')

    def enable_synchronization(self, enable: bool):
        """Enable or disable synchronization"""
        self.synchronization_enabled = enable
        self.get_logger().info(f'Synchronization {"enabled" if enable else "disabled"}')

    def get_sync_statistics(self) -> Tuple[float, float, float]:
        """Get synchronization statistics (avg error, max error, time offset)"""
        return self.avg_sync_error, self.max_sync_error, self.time_offset

    def get_state_difference(self) -> Dict[str, float]:
        """Get the difference between physical and virtual states"""
        if not self.physical_state or not self.virtual_state:
            return {}

        diff = {}
        common_joints = set(self.physical_state.joint_positions.keys()) & set(self.virtual_state.joint_positions.keys())

        for joint in common_joints:
            phys_pos = self.physical_state.joint_positions[joint]
            virt_pos = self.virtual_state.joint_positions[joint]
            diff[joint] = abs(phys_pos - virt_pos)

        return diff

    def force_resync(self):
        """Force resynchronization between physical and virtual states"""
        with self.state_lock:
            if self.physical_state and self.virtual_state:
                # Copy physical state to virtual state completely
                self.virtual_state.joint_positions = dict(self.physical_state.joint_positions)
                self.virtual_state.joint_velocities = dict(self.physical_state.joint_velocities)
                self.virtual_state.joint_efforts = dict(self.physical_state.joint_efforts)
                self.virtual_state.position = self.physical_state.position
                self.virtual_state.orientation = self.physical_state.orientation
                self.virtual_state.linear_velocity = self.physical_state.linear_velocity
                self.virtual_state.angular_velocity = self.physical_state.angular_velocity
                self.virtual_state.timestamp = self.physical_state.timestamp

                self.get_logger().info('Forced resynchronization completed')


def main(args=None):
    rclpy.init(args=args)

    synchronization_node = SynchronizationNode()

    try:
        rclpy.spin(synchronization_node)
    except KeyboardInterrupt:
        pass
    finally:
        synchronization_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()