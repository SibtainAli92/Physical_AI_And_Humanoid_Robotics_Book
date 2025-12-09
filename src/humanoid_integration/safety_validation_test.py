#!/usr/bin/env python3
# safety_validation_test.py
# Safety validation and compliance testing script

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool, Float64
from builtin_interfaces.msg import Duration
import numpy as np
import time
import csv
from datetime import datetime
from typing import Dict, List, Tuple
import threading

class SafetyValidationTester(Node):
    """
    Safety validation and compliance testing system
    """
    def __init__(self):
        super().__init__('safety_validation_tester')

        # Test configuration
        self.declare_parameter('test_duration', 30.0)  # seconds
        self.declare_parameter('test_sample_rate', 100.0)  # Hz
        self.declare_parameter('collision_force_threshold', 50.0)  # N
        self.declare_parameter('max_angular_velocity', 3.0)  # rad/s
        self.declare_parameter('max_linear_acceleration', 20.0)  # m/s^2
        self.declare_parameter('fall_angle_threshold', 0.5)  # rad

        self.test_duration = self.get_parameter('test_duration').value
        self.test_sample_rate = self.get_parameter('test_sample_rate').value
        self.collision_force_threshold = self.get_parameter('collision_force_threshold').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.max_linear_acceleration = self.get_parameter('max_linear_acceleration').value
        self.fall_angle_threshold = self.get_parameter('fall_angle_threshold').value

        # Publishers and subscribers for monitoring
        qos_profile = QoSProfile(depth=10)

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            qos_profile
        )

        self.ft_sub = self.create_subscription(
            WrenchStamped,
            '/left_foot/ft_sensor',
            self.ft_callback,
            qos_profile
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_stop_callback,
            qos_profile
        )

        # Initialize data storage
        self.joint_data = []
        self.imu_data = []
        self.ft_data = []
        self.emergency_stop_events = []

        # Test state
        self.test_running = False
        self.test_start_time = None
        self.test_results = {}

        self.get_logger().info('Safety Validation Tester initialized')

    def joint_state_callback(self, msg: JointState):
        """
        Callback for joint state messages during testing
        """
        if self.test_running:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.joint_data.append({
                'timestamp': timestamp,
                'names': msg.name,
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort)
            })

    def imu_callback(self, msg: Imu):
        """
        Callback for IMU messages during testing
        """
        if self.test_running:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.imu_data.append({
                'timestamp': timestamp,
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            })

    def ft_callback(self, msg: WrenchStamped):
        """
        Callback for force/torque messages during testing
        """
        if self.test_running:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.ft_data.append({
                'timestamp': timestamp,
                'force': [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                'torque': [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
            })

    def emergency_stop_callback(self, msg: Bool):
        """
        Callback for emergency stop events
        """
        if self.test_running:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.emergency_stop_events.append({
                'timestamp': timestamp,
                'activated': msg.data
            })

    def start_test(self, test_name: str):
        """
        Start a safety validation test
        """
        self.get_logger().info(f'Starting safety validation test: {test_name}')

        self.test_running = True
        self.test_start_time = self.get_clock().now().nanoseconds / 1e9

        # Clear previous test data
        self.joint_data = []
        self.imu_data = []
        self.ft_data = []
        self.emergency_stop_events = []

        # Run the test for specified duration
        test_timer = self.create_timer(self.test_duration, self.stop_test)

        return test_timer

    def stop_test(self):
        """
        Stop the current safety validation test
        """
        self.test_running = False
        self.get_logger().info('Safety validation test stopped')

        # Process test results
        self.process_test_results()

    def process_test_results(self):
        """
        Process and analyze the collected test data
        """
        self.get_logger().info('Processing test results...')

        # Analyze joint data
        joint_analysis = self.analyze_joint_data()

        # Analyze IMU data
        imu_analysis = self.analyze_imu_data()

        # Analyze force/torque data
        ft_analysis = self.analyze_ft_data()

        # Analyze emergency stop events
        emergency_analysis = self.analyze_emergency_events()

        # Compile results
        self.test_results = {
            'joint_analysis': joint_analysis,
            'imu_analysis': imu_analysis,
            'ft_analysis': ft_analysis,
            'emergency_analysis': emergency_analysis,
            'compliance_status': self.calculate_compliance_status()
        }

        self.get_logger().info('Test results processed')
        self.print_test_summary()

    def analyze_joint_data(self) -> Dict:
        """
        Analyze joint state data for safety compliance
        """
        if not self.joint_data:
            return {'status': 'No data', 'violations': 0}

        # Check velocity limits
        velocity_violations = 0
        max_velocity = 0.0

        for data_point in self.joint_data:
            for vel in data_point['velocities']:
                if abs(vel) > max_velocity:
                    max_velocity = abs(vel)
                if abs(vel) > self.max_angular_velocity:
                    velocity_violations += 1

        # Check effort limits
        effort_violations = 0
        max_effort = 0.0

        for data_point in self.joint_data:
            for eff in data_point['efforts']:
                if abs(eff) > max_effort:
                    max_effort = abs(eff)
                if abs(eff) > 100.0:  # Assuming max effort of 100N for this test
                    effort_violations += 1

        return {
            'status': 'Analyzed',
            'velocity_violations': velocity_violations,
            'max_velocity': max_velocity,
            'effort_violations': effort_violations,
            'max_effort': max_effort
        }

    def analyze_imu_data(self) -> Dict:
        """
        Analyze IMU data for safety compliance
        """
        if not self.imu_data:
            return {'status': 'No data', 'violations': 0}

        # Check angular velocity limits
        angular_velocity_violations = 0
        max_angular_velocity = 0.0

        for data_point in self.imu_data:
            ang_vel = np.sqrt(sum([v**2 for v in data_point['angular_velocity']]))
            if ang_vel > max_angular_velocity:
                max_angular_velocity = ang_vel
            if ang_vel > self.max_angular_velocity:
                angular_velocity_violations += 1

        # Check linear acceleration limits
        linear_acceleration_violations = 0
        max_linear_acceleration = 0.0

        for data_point in self.imu_data:
            lin_acc = np.sqrt(sum([a**2 for a in data_point['linear_acceleration']]))
            if lin_acc > max_linear_acceleration:
                max_linear_acceleration = lin_acc
            if lin_acc > self.max_linear_acceleration:
                linear_acceleration_violations += 1

        # Check for potential falls
        fall_events = 0
        for data_point in self.imu_data:
            quat = data_point['orientation']
            roll, pitch, _ = self.quaternion_to_euler(quat)
            if abs(roll) > self.fall_angle_threshold or abs(pitch) > self.fall_angle_threshold:
                fall_events += 1

        return {
            'status': 'Analyzed',
            'angular_velocity_violations': angular_velocity_violations,
            'max_angular_velocity': max_angular_velocity,
            'linear_acceleration_violations': linear_acceleration_violations,
            'max_linear_acceleration': max_linear_acceleration,
            'potential_fall_events': fall_events
        }

    def analyze_ft_data(self) -> Dict:
        """
        Analyze force/torque sensor data for safety compliance
        """
        if not self.ft_data:
            return {'status': 'No data', 'violations': 0}

        # Check force limits
        force_violations = 0
        max_force = 0.0

        for data_point in self.ft_data:
            force_magnitude = np.sqrt(sum([f**2 for f in data_point['force']]))
            if force_magnitude > max_force:
                max_force = force_magnitude
            if force_magnitude > self.collision_force_threshold:
                force_violations += 1

        return {
            'status': 'Analyzed',
            'force_violations': force_violations,
            'max_force': max_force
        }

    def analyze_emergency_events(self) -> Dict:
        """
        Analyze emergency stop events
        """
        activation_count = sum(1 for event in self.emergency_stop_events if event['activated'])
        deactivation_count = sum(1 for event in self.emergency_stop_events if not event['activated'])

        return {
            'activation_count': activation_count,
            'deactivation_count': deactivation_count,
            'total_events': len(self.emergency_stop_events)
        }

    def calculate_compliance_status(self) -> str:
        """
        Calculate overall compliance status based on all analyses
        """
        joint_analysis = self.test_results.get('joint_analysis', {})
        imu_analysis = self.test_results.get('imu_analysis', {})
        ft_analysis = self.test_results.get('ft_analysis', {})

        # Check for any violations
        total_violations = (
            joint_analysis.get('velocity_violations', 0) +
            joint_analysis.get('effort_violations', 0) +
            imu_analysis.get('angular_velocity_violations', 0) +
            imu_analysis.get('linear_acceleration_violations', 0) +
            ft_analysis.get('force_violations', 0)
        )

        if total_violations == 0:
            return 'PASS'
        else:
            return 'FAIL'

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

    def print_test_summary(self):
        """
        Print a summary of the test results
        """
        print("\n" + "="*60)
        print("SAFETY VALIDATION TEST RESULTS SUMMARY")
        print("="*60)

        print(f"Compliance Status: {self.test_results.get('compliance_status', 'UNKNOWN')}")

        joint_analysis = self.test_results.get('joint_analysis', {})
        print(f"\nJoint Analysis:")
        print(f"  Velocity Violations: {joint_analysis.get('velocity_violations', 0)}")
        print(f"  Max Velocity Recorded: {joint_analysis.get('max_velocity', 0):.3f} rad/s")
        print(f"  Effort Violations: {joint_analysis.get('effort_violations', 0)}")
        print(f"  Max Effort Recorded: {joint_analysis.get('max_effort', 0):.3f} Nm")

        imu_analysis = self.test_results.get('imu_analysis', {})
        print(f"\nIMU Analysis:")
        print(f"  Angular Velocity Violations: {imu_analysis.get('angular_velocity_violations', 0)}")
        print(f"  Max Angular Velocity: {imu_analysis.get('max_angular_velocity', 0):.3f} rad/s")
        print(f"  Linear Acceleration Violations: {imu_analysis.get('linear_acceleration_violations', 0)}")
        print(f"  Max Linear Acceleration: {imu_analysis.get('max_linear_acceleration', 0):.3f} m/s²")
        print(f"  Potential Fall Events: {imu_analysis.get('potential_fall_events', 0)}")

        ft_analysis = self.test_results.get('ft_analysis', {})
        print(f"\nForce/Torque Analysis:")
        print(f"  Force Violations: {ft_analysis.get('force_violations', 0)}")
        print(f"  Max Force Recorded: {ft_analysis.get('max_force', 0):.3f} N")

        emergency_analysis = self.test_results.get('emergency_analysis', {})
        print(f"\nEmergency Stop Events:")
        print(f"  Activations: {emergency_analysis.get('activation_count', 0)}")
        print(f"  Deactivations: {emergency_analysis.get('deactivation_count', 0)}")

        print("="*60)

    def save_test_results(self, filename: str = None):
        """
        Save test results to a CSV file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"safety_test_results_{timestamp}.csv"

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'test_type', 'parameter', 'value', 'limit', 'status'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write joint analysis results
            joint_analysis = self.test_results.get('joint_analysis', {})
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Joint',
                'parameter': 'Velocity Violations',
                'value': joint_analysis.get('velocity_violations', 0),
                'limit': 0,
                'status': 'FAIL' if joint_analysis.get('velocity_violations', 0) > 0 else 'PASS'
            })

            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Joint',
                'parameter': 'Max Velocity',
                'value': joint_analysis.get('max_velocity', 0),
                'limit': self.max_angular_velocity,
                'status': 'FAIL' if joint_analysis.get('max_velocity', 0) > self.max_angular_velocity else 'PASS'
            })

            # Write IMU analysis results
            imu_analysis = self.test_results.get('imu_analysis', {})
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'test_type': 'IMU',
                'parameter': 'Angular Velocity Violations',
                'value': imu_analysis.get('angular_velocity_violations', 0),
                'limit': 0,
                'status': 'FAIL' if imu_analysis.get('angular_velocity_violations', 0) > 0 else 'PASS'
            })

            # Write force/torque analysis results
            ft_analysis = self.test_results.get('ft_analysis', {})
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Force/Torque',
                'parameter': 'Force Violations',
                'value': ft_analysis.get('force_violations', 0),
                'limit': 0,
                'status': 'FAIL' if ft_analysis.get('force_violations', 0) > 0 else 'PASS'
            })

        self.get_logger().info(f'Test results saved to {filename}')


def main(args=None):
    rclpy.init(args=args)

    tester = SafetyValidationTester()

    # Run a basic safety validation test
    tester.start_test("Emergency Stop and Safety Limits Validation")

    try:
        # Run for the duration of the test plus a little extra time
        test_duration = tester.test_duration + 5.0
        start_time = time.time()

        while time.time() - start_time < test_duration:
            rclpy.spin_once(tester, timeout_sec=0.1)

        # Save results
        tester.save_test_results()

    except KeyboardInterrupt:
        tester.get_logger().info('Test interrupted by user')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()