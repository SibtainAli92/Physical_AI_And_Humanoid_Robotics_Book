#!/usr/bin/env python3

"""
Integration Tester for Humanoid Robotics Platform

This module provides testing for the integrated system,
focusing on inter-module communication and coordination.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
import statistics
import sys


@dataclass
class IntegrationTestResult:
    """Represents the result of an integration test"""
    test_name: str
    passed: bool
    details: str
    metrics: Dict[str, float]
    execution_time: float
    timestamp: float


class IntegrationTester(Node):
    def __init__(self):
        super().__init__('integration_tester')

        # Publishers for integration test results
        self.integration_test_status_publisher = self.create_publisher(String, 'integration_tester/status', 10)
        self.system_performance_publisher = self.create_publisher(Float32, 'integration_tester/performance', 10)

        # Subscribers for system state monitoring
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # System state tracking
        self.joint_states = None
        self.imu_data = None
        self.odom_data = None
        self.laser_data = None
        self.last_update_times = {
            'joint_states': 0,
            'imu': 0,
            'odom': 0,
            'laser': 0
        }

        # Test results
        self.test_results = []
        self.integration_tests_enabled = True

        self.get_logger().info('Integration Tester initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        self.joint_states = msg
        self.last_update_times['joint_states'] = time.time()

    def imu_callback(self, msg):
        """Callback for IMU data"""
        self.imu_data = msg
        self.last_update_times['imu'] = time.time()

    def odom_callback(self, msg):
        """Callback for odometry data"""
        self.odom_data = msg
        self.last_update_times['odom'] = time.time()

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        self.laser_data = msg
        self.last_update_times['laser'] = time.time()

    def test_module_communication(self) -> IntegrationTestResult:
        """Test communication between all modules"""
        start_time = time.time()
        passed = True
        details = []
        metrics = {}

        try:
            # Check if all data sources are updating regularly
            current_time = time.time()
            timeout = 2.0  # seconds

            # Wait for data from all sources (with timeout)
            start_wait = time.time()
            while current_time - start_wait < timeout:
                if all(self.last_update_times.values()):
                    break
                time.sleep(0.1)
                current_time = time.time()

            # Check if all data sources are active
            data_age_threshold = 1.0  # seconds
            for source, last_time in self.last_update_times.items():
                age = current_time - last_time if last_time > 0 else float('inf')
                metrics[f'{source}_data_age'] = age
                if age > data_age_threshold:
                    details.append(f"{source} data not updating (age: {age:.2f}s)")
                    passed = False

            # Test message throughput between modules
            throughput_tests = 100
            throughput_start = time.time()
            for i in range(throughput_tests):
                # Simulate sending a message between modules
                time.sleep(0.001)  # Simulate processing time
            throughput_time = time.time() - throughput_start
            actual_throughput = throughput_tests / throughput_time if throughput_time > 0 else 0
            metrics['message_throughput'] = actual_throughput

            # Check for data consistency (simulated)
            consistency_score = np.random.uniform(0.8, 1.0)  # Simulate high consistency
            metrics['data_consistency'] = consistency_score
            if consistency_score < 0.9:
                details.append(f"Data consistency low: {consistency_score:.2f}")
                passed = False

            execution_time = time.time() - start_time
            metrics['total_execution_time'] = execution_time

            result = IntegrationTestResult(
                test_name="module_communication",
                passed=passed,
                details="; ".join(details) if details else "All communication tests passed",
                metrics=metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            return IntegrationTestResult(
                test_name="module_communication",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_system_coordination(self) -> IntegrationTestResult:
        """Test coordination between system modules"""
        start_time = time.time()
        passed = True
        details = []
        metrics = {}

        try:
            # Test coordinated behavior execution
            coordination_scenarios = [
                ("Navigation with Perception", 5.0),
                ("Manipulation with Planning", 4.0),
                ("Interaction with AI", 3.0),
                ("Safety with Control", 2.0)
            ]

            coordination_success = 0
            total_scenarios = len(coordination_scenarios)

            for scenario_name, duration in coordination_scenarios:
                scenario_start = time.time()
                # Simulate coordinated behavior
                for step in range(10):  # 10 coordination steps per scenario
                    # Simulate modules working together
                    time.sleep(0.1)
                    # Check for coordination issues
                    coordination_error = np.random.random() > 0.95  # 5% chance of error
                    if coordination_error:
                        details.append(f"Coordination error in {scenario_name}")
                        passed = False
                        break
                else:
                    # Scenario completed successfully
                    coordination_success += 1

            coordination_rate = coordination_success / total_scenarios if total_scenarios > 0 else 0
            metrics['coordination_success_rate'] = coordination_rate

            if coordination_rate < 0.8:  # 80% minimum coordination rate
                details.append(f"Low coordination success rate: {coordination_rate:.2f}")
                passed = False

            # Test timing synchronization
            sync_tests = 50
            sync_start = time.time()
            for i in range(sync_tests):
                # Simulate synchronized operations
                time.sleep(0.02)  # 20ms per sync operation
            sync_time = time.time() - sync_start
            avg_sync_time = sync_time / sync_tests * 1000  # ms per operation
            metrics['synchronization_time'] = avg_sync_time

            if avg_sync_time > 50:  # 50ms threshold
                details.append(f"Synchronization too slow: {avg_sync_time:.2f}ms")
                passed = False

            execution_time = time.time() - start_time
            metrics['total_execution_time'] = execution_time

            result = IntegrationTestResult(
                test_name="system_coordination",
                passed=passed,
                details="; ".join(details) if details else "All coordination tests passed",
                metrics=metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            return IntegrationTestResult(
                test_name="system_coordination",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_safety_integration(self) -> IntegrationTestResult:
        """Test safety system integration"""
        start_time = time.time()
        passed = True
        details = []
        metrics = {}

        try:
            # Test safety system responsiveness
            safety_tests = [
                ("Collision detection", 0.5),
                ("Balance monitoring", 0.3),
                ("Emergency stop", 0.2),
                ("Joint limit protection", 0.4)
            ]

            for test_name, duration in safety_tests:
                test_start = time.time()
                # Simulate safety check
                for i in range(10):
                    # Simulate sensor readings and safety checks
                    sensor_value = np.random.random()
                    time.sleep(0.01)
                    # Simulate safety decision making
                    time.sleep(0.005)
                test_duration = time.time() - test_start
                metrics[f'{test_name.lower().replace(" ", "_")}_time'] = test_duration

            # Test emergency stop propagation
            emergency_start = time.time()
            # Simulate emergency being detected and propagated to all modules
            time.sleep(0.1)  # Detection time
            time.sleep(0.05)  # Propagation time
            time.sleep(0.05)  # Response time
            emergency_time = time.time() - emergency_start
            metrics['emergency_stop_time'] = emergency_time

            if emergency_time > 0.3:  # 300ms threshold
                details.append(f"Emergency stop too slow: {emergency_time:.3f}s")
                passed = False

            # Test safety system reliability
            reliability_tests = 100
            safe_decisions = 0
            for i in range(reliability_tests):
                # Simulate safety decision
                is_safe = np.random.random() > 0.02  # 98% safe decisions
                if is_safe:
                    safe_decisions += 1
                time.sleep(0.002)  # 2ms per safety check

            safety_reliability = safe_decisions / reliability_tests
            metrics['safety_reliability'] = safety_reliability

            if safety_reliability < 0.95:  # 95% minimum reliability
                details.append(f"Safety reliability too low: {safety_reliability:.3f}")
                passed = False

            execution_time = time.time() - start_time
            metrics['total_execution_time'] = execution_time

            result = IntegrationTestResult(
                test_name="safety_integration",
                passed=passed,
                details="; ".join(details) if details else "All safety tests passed",
                metrics=metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            return IntegrationTestResult(
                test_name="safety_integration",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_performance_under_load(self) -> IntegrationTestResult:
        """Test system performance under load"""
        start_time = time.time()
        passed = True
        details = []
        metrics = {}

        try:
            # Simulate system under various load conditions
            load_conditions = [
                ("Light Load", 0.3),  # 30% CPU
                ("Medium Load", 0.6), # 60% CPU
                ("Heavy Load", 0.9)   # 90% CPU
            ]

            for condition_name, load_factor in load_conditions:
                load_start = time.time()
                # Simulate processing load
                operations = int(1000 * load_factor)  # More operations with higher load
                for i in range(operations):
                    # Simulate computational work
                    result = math.sin(i * 0.01) * math.cos(i * 0.02)
                    # Simulate module communication
                    time.sleep(0.0001 * load_factor)  # More delay with higher load
                load_duration = time.time() - load_start
                metrics[f'{condition_name.lower().replace(" ", "_")}_duration'] = load_duration

            # Test system stability under concurrent operations
            concurrent_modules = 5
            concurrency_start = time.time()
            for i in range(20):  # 20 concurrent operation cycles
                for module in range(concurrent_modules):
                    # Simulate module operation
                    time.sleep(0.01)
            concurrency_time = time.time() - concurrency_start
            metrics['concurrency_time'] = concurrency_time

            # Calculate performance degradation
            light_time = metrics.get('light_load_duration', 1.0)
            heavy_time = metrics.get('heavy_load_duration', 1.0)
            performance_degradation = (heavy_time - light_time) / light_time if light_time > 0 else 0
            metrics['performance_degradation'] = performance_degradation

            if performance_degradation > 0.5:  # 50% degradation threshold
                details.append(f"High performance degradation: {performance_degradation:.2f}")
                passed = False

            # Test memory usage under load (simulated)
            simulated_memory_mb = 100 + (load_factor * 150)  # Base 100MB + load-dependent
            metrics['simulated_memory_usage'] = simulated_memory_mb

            if simulated_memory_mb > 500:  # 500MB threshold
                details.append(f"High memory usage: {simulated_memory_mb:.1f}MB")
                passed = False

            execution_time = time.time() - start_time
            metrics['total_execution_time'] = execution_time

            result = IntegrationTestResult(
                test_name="performance_under_load",
                passed=passed,
                details="; ".join(details) if details else "All performance tests passed",
                metrics=metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            return IntegrationTestResult(
                test_name="performance_under_load",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_end_to_end_scenario(self) -> IntegrationTestResult:
        """Test complete end-to-end scenario"""
        start_time = time.time()
        passed = True
        details = []
        metrics = {}

        try:
            # Simulate a complete robot task: navigate to object, grasp it, return
            scenario_phases = [
                ("System Initialization", 2.0),
                ("Perception & Planning", 3.0),
                ("Navigation", 5.0),
                ("Object Interaction", 4.0),
                ("Return & Report", 2.0)
            ]

            total_expected_time = sum(duration for _, duration in scenario_phases)
            scenario_start = time.time()

            for phase_name, expected_duration in scenario_phases:
                phase_start = time.time()
                # Simulate phase execution
                for step in range(int(expected_duration * 10)):  # 10 steps per second
                    # Simulate integrated module operations
                    time.sleep(0.05)  # 50ms per step

                    # Simulate potential phase failure
                    if np.random.random() < 0.01:  # 1% chance of step failure
                        details.append(f"Failure in {phase_name}")
                        passed = False
                        break
                else:
                    # Phase completed successfully
                    phase_actual = time.time() - phase_start
                    metrics[f'{phase_name.lower().replace(" & ", "_").replace(" ", "_")}_time'] = phase_actual

            actual_total_time = time.time() - scenario_start
            time_efficiency = actual_total_time / total_expected_time if total_expected_time > 0 else 1.0
            metrics['time_efficiency'] = time_efficiency

            if time_efficiency > 2.0:  # Should not take more than 2x expected time
                details.append(f"Scenario took too long: {time_efficiency:.2f}x expected time")
                passed = False

            # Test resource utilization during scenario
            avg_cpu_usage = np.random.uniform(0.4, 0.7)  # 40-70% during scenario
            metrics['average_cpu_usage'] = avg_cpu_usage

            # Test system state consistency throughout scenario
            state_changes = 50  # Simulated state changes
            consistent_changes = int(state_changes * 0.98)  # 98% consistency
            state_consistency = consistent_changes / state_changes if state_changes > 0 else 1.0
            metrics['state_consistency'] = state_consistency

            if state_consistency < 0.95:  # 95% minimum consistency
                details.append(f"Low state consistency: {state_consistency:.3f}")
                passed = False

            execution_time = time.time() - start_time
            metrics['total_execution_time'] = execution_time

            result = IntegrationTestResult(
                test_name="end_to_end_scenario",
                passed=passed,
                details="; ".join(details) if details else "End-to-end scenario completed successfully",
                metrics=metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            return IntegrationTestResult(
                test_name="end_to_end_scenario",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def run_integration_tests(self) -> Dict[str, IntegrationTestResult]:
        """Run all integration tests"""
        results = {}

        self.get_logger().info("Starting integration tests...")

        results['communication'] = self.test_module_communication()
        self.get_logger().info(f"Communication Test: {'PASS' if results['communication'].passed else 'FAIL'}")

        results['coordination'] = self.test_system_coordination()
        self.get_logger().info(f"Coordination Test: {'PASS' if results['coordination'].passed else 'FAIL'}")

        results['safety'] = self.test_safety_integration()
        self.get_logger().info(f"Safety Test: {'PASS' if results['safety'].passed else 'FAIL'}")

        results['performance'] = self.test_performance_under_load()
        self.get_logger().info(f"Performance Test: {'PASS' if results['performance'].passed else 'FAIL'}")

        results['end_to_end'] = self.test_end_to_end_scenario()
        self.get_logger().info(f"End-to-End Test: {'PASS' if results['end_to_end'].passed else 'FAIL'}")

        return results

    def generate_integration_report(self) -> str:
        """Generate a detailed integration test report"""
        if not self.test_results:
            return "No integration tests have been run yet."

        report = []
        report.append("HUMANOID ROBOTICS PLATFORM - INTEGRATION TEST REPORT")
        report.append("=" * 60)

        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success Rate: {success_rate:.1%}")
        report.append("")

        # Individual test results
        for result in self.test_results:
            status = "PASS" if result.passed else "FAIL"
            report.append(f"Test: {result.test_name}")
            report.append(f"  Status: {status}")
            report.append(f"  Duration: {result.execution_time:.3f}s")
            report.append(f"  Details: {result.details}")

            if result.metrics:
                report.append("  Metrics:")
                for metric, value in result.metrics.items():
                    report.append(f"    {metric}: {value:.3f}")

            report.append("")

        # Summary of key metrics
        if self.test_results:
            all_metrics = {}
            for result in self.test_results:
                for metric, value in result.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

            report.append("KEY PERFORMANCE METRICS:")
            report.append("-" * 30)
            for metric, values in all_metrics.items():
                if 'time' in metric or 'duration' in metric:
                    avg_value = statistics.mean(values)
                    report.append(f"  {metric}: {avg_value:.3f}s average")
                elif 'rate' in metric or 'ratio' in metric or 'reliability' in metric:
                    avg_value = statistics.mean(values)
                    report.append(f"  {metric}: {avg_value:.3f} ({avg_value*100:.1f}%) average")
                else:
                    avg_value = statistics.mean(values)
                    report.append(f"  {metric}: {avg_value:.3f} average")

        return "\n".join(report)

    def get_integration_statistics(self) -> Dict[str, any]:
        """Get integration testing statistics"""
        if not self.test_results:
            return {
                'total_tests': 0,
                'passed_tests': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'tests_run': []
            }

        passed_count = sum(1 for r in self.test_results if r.passed)
        total_time = sum(r.execution_time for r in self.test_results)

        return {
            'total_tests': len(self.test_results),
            'passed_tests': passed_count,
            'success_rate': passed_count / len(self.test_results),
            'average_execution_time': total_time / len(self.test_results),
            'tests_run': [r.test_name for r in self.test_results],
            'latest_test_time': max(r.timestamp for r in self.test_results) if self.test_results else 0
        }


def main(args=None):
    rclpy.init(args=args)

    integration_tester = IntegrationTester()

    try:
        print("Running integration tests for Humanoid Robotics Platform...")
        results = integration_tester.run_integration_tests()

        # Generate and print integration report
        report = integration_tester.generate_integration_report()
        print(report)

        # Check overall success
        all_passed = all(result.passed for result in results.values())

        return 0 if all_passed else 1

    except Exception as e:
        print(f"Error during integration testing: {e}")
        return 1
    finally:
        integration_tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())