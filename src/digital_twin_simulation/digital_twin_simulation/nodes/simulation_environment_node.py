#!/usr/bin/env python3

"""
Simulation Environment Node for Humanoid Robot Digital Twin

This node orchestrates the digital twin simulation environment,
managing the virtual world, physics simulation, and synchronization
with the physical robot system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Vector3, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import time
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RobotState:
    """Data class to represent robot state in simulation"""
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]
    joint_efforts: Dict[str, float]
    position: Point
    orientation: Vector3
    linear_velocity: Vector3
    angular_velocity: Vector3
    timestamp: float


class SimulationEnvironmentNode(Node):
    def __init__(self):
        super().__init__('simulation_environment_node')

        # Publishers for simulation data
        self.sim_robot_state_publisher = self.create_publisher(JointState, 'sim/joint_states', 10)
        self.sim_imu_publisher = self.create_publisher(Imu, 'sim/imu/data', 10)
        self.sim_laser_publisher = self.create_publisher(LaserScan, 'sim/scan', 10)
        self.sim_odom_publisher = self.create_publisher(Odometry, 'sim/odom', 10)
        self.sim_environment_publisher = self.create_publisher(MarkerArray, 'sim/environment', 10)
        self.sim_status_publisher = self.create_publisher(String, 'sim/status', 10)

        # Subscribers for real robot data (for synchronization)
        self.real_joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.real_joint_state_callback, 10)
        self.real_odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.real_odom_callback, 10)

        # Timer for simulation update
        self.sim_timer = self.create_timer(0.016, self.simulation_update_callback)  # ~60 Hz

        # Simulation state
        self.sim_time = time.time()
        self.real_robot_state: Optional[RobotState] = None
        self.sim_robot_state: Optional[RobotState] = None
        self.environment_objects = []
        self.physics_enabled = True
        self.synchronization_mode = 'mirror'  # 'mirror', 'predict', 'independent'

        # Initialize robot state
        self.initialize_robot_state()

        # Simulation parameters
        self.gravity = -9.81  # m/s^2
        self.simulation_speed = 1.0  # 1x real-time speed
        self.collision_detection = True

        self.get_logger().info('Simulation Environment Node initialized')

    def initialize_robot_state(self):
        """Initialize the robot state for simulation"""
        # Default joint positions for humanoid robot
        default_joints = {
            'hip_joint_left': 0.0,
            'hip_joint_right': 0.0,
            'knee_joint_left': 0.0,
            'knee_joint_right': 0.0,
            'ankle_joint_left': 0.0,
            'ankle_joint_right': 0.0,
            'shoulder_joint_left': 0.0,
            'shoulder_joint_right': 0.0,
            'elbow_joint_left': 0.0,
            'elbow_joint_right': 0.0,
        }

        # Initialize real robot state
        self.real_robot_state = RobotState(
            joint_positions=default_joints.copy(),
            joint_velocities={k: 0.0 for k in default_joints.keys()},
            joint_efforts={k: 0.0 for k in default_joints.keys()},
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Vector3(x=0.0, y=0.0, z=0.0),
            linear_velocity=Vector3(x=0.0, y=0.0, z=0.0),
            angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
            timestamp=time.time()
        )

        # Initialize sim robot state (copy of real initially)
        self.sim_robot_state = RobotState(
            joint_positions=default_joints.copy(),
            joint_velocities={k: 0.0 for k in default_joints.keys()},
            joint_efforts={k: 0.0 for k in default_joints.keys()},
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Vector3(x=0.0, y=0.0, z=0.0),
            linear_velocity=Vector3(x=0.0, y=0.0, z=0.0),
            angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
            timestamp=time.time()
        )

    def real_joint_state_callback(self, msg):
        """Callback for real robot joint state updates"""
        if self.synchronization_mode == 'mirror':
            # Update sim state to match real robot
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.sim_robot_state.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.sim_robot_state.joint_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.sim_robot_state.joint_efforts[name] = msg.effort[i]
        else:
            # In other modes, just store the real state for comparison
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.real_robot_state.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.real_robot_state.joint_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.real_robot_state.joint_efforts[name] = msg.effort[i]

        self.real_robot_state.timestamp = time.time()

    def real_odom_callback(self, msg):
        """Callback for real robot odometry updates"""
        if self.synchronization_mode == 'mirror':
            # Update sim position to match real robot
            self.sim_robot_state.position.x = msg.pose.pose.position.x
            self.sim_robot_state.position.y = msg.pose.pose.position.y
            self.sim_robot_state.position.z = msg.pose.pose.position.z

            # Convert quaternion to Euler angles for orientation
            # (Simplified - in reality would keep quaternion)
            self.sim_robot_state.orientation.x = msg.pose.pose.orientation.x
            self.sim_robot_state.orientation.y = msg.pose.pose.orientation.y
            self.sim_robot_state.orientation.z = msg.pose.pose.orientation.z
        else:
            # Store real odometry for comparison
            self.real_robot_state.position.x = msg.pose.pose.position.x
            self.real_robot_state.position.y = msg.pose.pose.position.y
            self.real_robot_state.position.z = msg.pose.pose.position.z
            self.real_robot_state.orientation.x = msg.pose.pose.orientation.x
            self.real_robot_state.orientation.y = msg.pose.pose.orientation.y
            self.real_robot_state.orientation.z = msg.pose.pose.orientation.z

    def simulation_update_callback(self):
        """Main simulation update callback"""
        current_time = time.time()
        dt = current_time - self.sim_time
        self.sim_time = current_time

        # Update simulation based on mode
        if self.synchronization_mode == 'independent':
            self.update_independent_simulation(dt)
        elif self.synchronization_mode == 'predict':
            self.update_predictive_simulation(dt)
        else:  # mirror mode
            self.update_mirror_simulation(dt)

        # Apply physics if enabled
        if self.physics_enabled:
            self.apply_physics(dt)

        # Publish simulation data
        self.publish_simulation_data()

        # Publish environment visualization
        self.publish_environment()

        # Publish status
        status_msg = String()
        status_msg.data = f"SIM_RUNNING: {current_time:.2f}, Mode: {self.synchronization_mode}"
        self.sim_status_publisher.publish(status_msg)

    def update_independent_simulation(self, dt):
        """Update simulation in independent mode"""
        # Apply some default motion to make simulation interesting
        # This could be controlled by external commands in a real system
        for joint_name in self.sim_robot_state.joint_positions:
            # Simple oscillating motion for demonstration
            base_freq = 0.5  # Hz
            amplitude = 0.2
            self.sim_robot_state.joint_positions[joint_name] += amplitude * math.sin(base_freq * 2 * math.pi * self.sim_time) * dt

    def update_predictive_simulation(self, dt):
        """Update simulation in predictive mode"""
        # Predict future state based on current real robot state
        # This would use more sophisticated prediction algorithms in reality
        for joint_name in self.sim_robot_state.joint_positions:
            if joint_name in self.real_robot_state.joint_positions:
                # Simple prediction based on velocity
                current_pos = self.real_robot_state.joint_positions[joint_name]
                current_vel = self.real_robot_state.joint_velocities[joint_name]
                predicted_pos = current_pos + current_vel * dt * 1.1  # 10% prediction factor
                self.sim_robot_state.joint_positions[joint_name] = predicted_pos

    def update_mirror_simulation(self, dt):
        """Update simulation in mirror mode"""
        # In mirror mode, sim state is updated directly from real robot callbacks
        # This function exists to maintain consistent interface
        pass

    def apply_physics(self, dt):
        """Apply basic physics to the simulation"""
        # Apply gravity effect (simplified)
        self.sim_robot_state.linear_velocity.z += self.gravity * dt

        # Update position based on velocity
        self.sim_robot_state.position.x += self.sim_robot_state.linear_velocity.x * dt
        self.sim_robot_state.position.y += self.sim_robot_state.linear_velocity.y * dt
        self.sim_robot_state.position.z += self.sim_robot_state.linear_velocity.z * dt

        # Simple collision detection with ground (z=0)
        if self.sim_robot_state.position.z < 0:
            self.sim_robot_state.position.z = 0
            self.sim_robot_state.linear_velocity.z = 0

        # Apply damping to simulate friction
        damping_factor = 0.99
        self.sim_robot_state.linear_velocity.x *= damping_factor
        self.sim_robot_state.linear_velocity.y *= damping_factor
        self.sim_robot_state.angular_velocity.x *= damping_factor
        self.sim_robot_state.angular_velocity.y *= damping_factor
        self.sim_robot_state.angular_velocity.z *= damping_factor

    def publish_simulation_data(self):
        """Publish simulation data to ROS topics"""
        # Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = 'sim_base_link'

        for joint_name, position in self.sim_robot_state.joint_positions.items():
            joint_state_msg.name.append(joint_name)
            joint_state_msg.position.append(position)
            joint_state_msg.velocity.append(self.sim_robot_state.joint_velocities[joint_name])
            joint_state_msg.effort.append(self.sim_robot_state.joint_efforts[joint_name])

        self.sim_robot_state_publisher.publish(joint_state_msg)

        # Publish IMU data (simulated)
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'sim_imu_link'

        # Simulate IMU readings with some noise
        imu_msg.linear_acceleration.x = self.sim_robot_state.linear_velocity.x * 10 + (0.1 * (2 * (time.time() % 1) - 1))
        imu_msg.linear_acceleration.y = self.sim_robot_state.linear_velocity.y * 10 + (0.1 * (2 * (time.time() % 1) - 1))
        imu_msg.linear_acceleration.z = self.sim_robot_state.linear_velocity.z * 10 + self.gravity + (0.1 * (2 * (time.time() % 1) - 1))

        # Simulate angular velocity
        imu_msg.angular_velocity.x = self.sim_robot_state.angular_velocity.x
        imu_msg.angular_velocity.y = self.sim_robot_state.angular_velocity.y
        imu_msg.angular_velocity.z = self.sim_robot_state.angular_velocity.z

        # Simulate orientation (simplified)
        imu_msg.orientation.w = 1.0  # Simplified quaternion
        imu_msg.orientation.x = self.sim_robot_state.orientation.x
        imu_msg.orientation.y = self.sim_robot_state.orientation.y
        imu_msg.orientation.z = self.sim_robot_state.orientation.z

        self.sim_imu_publisher.publish(imu_msg)

        # Publish laser scan (simulated)
        laser_msg = LaserScan()
        laser_msg.header.stamp = self.get_clock().now().to_msg()
        laser_msg.header.frame_id = 'sim_laser_link'
        laser_msg.angle_min = -math.pi / 2
        laser_msg.angle_max = math.pi / 2
        laser_msg.angle_increment = math.pi / 180  # 1 degree
        laser_msg.time_increment = 0.0
        laser_msg.scan_time = 0.1
        laser_msg.range_min = 0.1
        laser_msg.range_max = 10.0

        # Simulate some ranges (in a simple environment)
        num_ranges = int((laser_msg.angle_max - laser_msg.angle_min) / laser_msg.angle_increment) + 1
        laser_msg.ranges = [8.0] * num_ranges  # Default to 8m (no obstacles nearby)

        # Add some variation to make it more realistic
        for i in range(len(laser_msg.ranges)):
            angle = laser_msg.angle_min + i * laser_msg.angle_increment
            # Simulate a wall at 2m in front
            if abs(angle) < 0.2:  # Front-facing sensors
                laser_msg.ranges[i] = 2.0 + 0.1 * (2 * (i % 2) - 1)  # Add small variation

        self.sim_laser_publisher.publish(laser_msg)

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'sim_odom'
        odom_msg.child_frame_id = 'sim_base_link'

        odom_msg.pose.pose.position = self.sim_robot_state.position
        odom_msg.pose.pose.orientation.w = 1.0  # Simplified
        odom_msg.pose.pose.orientation.x = self.sim_robot_state.orientation.x
        odom_msg.pose.pose.orientation.y = self.sim_robot_state.orientation.y
        odom_msg.pose.pose.orientation.z = self.sim_robot_state.orientation.z

        odom_msg.twist.twist.linear = self.sim_robot_state.linear_velocity
        odom_msg.twist.twist.angular = self.sim_robot_state.angular_velocity

        self.sim_odom_publisher.publish(odom_msg)

    def publish_environment(self):
        """Publish environment visualization markers"""
        marker_array = MarkerArray()
        marker_id = 0

        # Create ground plane marker
        ground_marker = Marker()
        ground_marker.header.frame_id = "sim_world"
        ground_marker.header.stamp = self.get_clock().now().to_msg()
        ground_marker.id = marker_id
        marker_id += 1
        ground_marker.type = Marker.CUBE
        ground_marker.action = Marker.ADD
        ground_marker.pose.position.x = 0.0
        ground_marker.pose.position.y = 0.0
        ground_marker.pose.position.z = -0.01  # Slightly below 0 to avoid z-fighting
        ground_marker.pose.orientation.w = 1.0
        ground_marker.scale.x = 20.0  # 20m x 20m area
        ground_marker.scale.y = 20.0
        ground_marker.scale.z = 0.02  # 2cm thick
        ground_marker.color.r = 0.5
        ground_marker.color.g = 0.5
        ground_marker.color.b = 0.5
        ground_marker.color.a = 0.8
        marker_array.markers.append(ground_marker)

        # Create some obstacle markers
        obstacles = [
            {'x': 2.0, 'y': 0.0, 'z': 0.5, 'size': [0.5, 0.5, 1.0]},
            {'x': -1.5, 'y': 1.0, 'z': 0.3, 'size': [0.3, 0.3, 0.6]},
            {'x': 0.0, 'y': -2.0, 'z': 0.4, 'size': [0.8, 0.4, 0.8]}
        ]

        for obs in obstacles:
            obs_marker = Marker()
            obs_marker.header.frame_id = "sim_world"
            obs_marker.header.stamp = self.get_clock().now().to_msg()
            obs_marker.id = marker_id
            marker_id += 1
            obs_marker.type = Marker.CUBE
            obs_marker.action = Marker.ADD
            obs_marker.pose.position.x = obs['x']
            obs_marker.pose.position.y = obs['y']
            obs_marker.pose.position.z = obs['z'] / 2  # Center at half height
            obs_marker.pose.orientation.w = 1.0
            obs_marker.scale.x = obs['size'][0]
            obs_marker.scale.y = obs['size'][1]
            obs_marker.scale.z = obs['size'][2]
            obs_marker.color.r = 0.8
            obs_marker.color.g = 0.2
            obs_marker.color.b = 0.2
            obs_marker.color.a = 0.8
            marker_array.markers.append(obs_marker)

        self.sim_environment_publisher.publish(marker_array)

    def set_synchronization_mode(self, mode):
        """Set the synchronization mode for the simulation"""
        if mode in ['mirror', 'predict', 'independent']:
            self.synchronization_mode = mode
            self.get_logger().info(f'Synchronization mode set to: {mode}')
        else:
            self.get_logger().warn(f'Invalid synchronization mode: {mode}')

    def enable_physics(self, enable):
        """Enable or disable physics simulation"""
        self.physics_enabled = enable
        self.get_logger().info(f'Physics simulation {"enabled" if enable else "disabled"}')


def main(args=None):
    rclpy.init(args=args)

    simulation_environment_node = SimulationEnvironmentNode()

    try:
        rclpy.spin(simulation_environment_node)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_environment_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()