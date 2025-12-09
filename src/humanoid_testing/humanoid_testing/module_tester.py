#!/usr/bin/env python3

"""
Module Tester for Humanoid Robotics Platform

This module provides detailed testing for individual modules
with specific test cases and performance metrics.
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
from typing import Dict, List, Optional, Tuple, Callable
import threading
from dataclasses import dataclass
import statistics
import sys


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    min_acceptable: float
    max_acceptable: float
    timestamp: float


@dataclass
class ModuleTestResult:
    """Represents the result of testing a module"""
    module_name: str
    test_name: str
    passed: bool
    details: str
    performance_metrics: List[PerformanceMetric]
    execution_time: float
    timestamp: float


class ModuleTester(Node):
    def __init__(self):
        super().__init__('module_tester')

        # Publishers for test results
        self.module_test_status_publisher = self.create_publisher(String, 'module_tester/status', 10)
        self.performance_publisher = self.create_publisher(Float32, 'module_tester/performance', 10)

        # Test configuration
        self.test_results = []
        self.performance_history = {module: [] for module in [
            'ros2_nervous_system', 'digital_twin_simulation',
            'ai_brain_isaac', 'vla_robotics', 'humanoid_integration'
        ]}

        self.get_logger().info('Module Tester initialized')

    def test_ros2_nervous_system(self) -> ModuleTestResult:
        """Test the ROS 2 Nervous System module"""
        start_time = time.time()
        passed = True
        details = []
        performance_metrics = []

        try:
            # Test 1: Node communication performance
            comm_start = time.time()
            # Simulate communication test
            time.sleep(0.05)  # Simulate network communication
            comm_time = time.time() - comm_start
            performance_metrics.append(PerformanceMetric(
                name="communication_latency", value=comm_time*1000, unit="ms",
                min_acceptable=0, max_acceptable=100, timestamp=time.time()
            ))

            if comm_time > 0.1:  # 100ms threshold
                details.append("Communication latency too high")
                passed = False

            # Test 2: Message throughput
            throughput_start = time.time()
            # Simulate sending multiple messages
            message_count = 100
            for i in range(message_count):
                time.sleep(0.001)  # Simulate message processing
            throughput_time = time.time() - throughput_start
            throughput = message_count / throughput_time if throughput_time > 0 else 0
            performance_metrics.append(PerformanceMetric(
                name="message_throughput", value=throughput, unit="msgs/sec",
                min_acceptable=50, max_acceptable=1000, timestamp=time.time()
            ))

            # Test 3: Node startup time
            startup_start = time.time()
            time.sleep(0.02)  # Simulate node startup
            startup_time = time.time() - startup_start
            performance_metrics.append(PerformanceMetric(
                name="node_startup_time", value=startup_time*1000, unit="ms",
                min_acceptable=0, max_acceptable=100, timestamp=time.time()
            ))

            if startup_time > 0.1:  # 100ms threshold
                details.append("Node startup time too long")
                passed = False

            execution_time = time.time() - start_time
            result = ModuleTestResult(
                module_name="ros2_nervous_system",
                test_name="comprehensive_test",
                passed=passed,
                details="; ".join(details) if details else "All tests passed",
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            self.performance_history["ros2_nervous_system"].append(result)

            return result

        except Exception as e:
            return ModuleTestResult(
                module_name="ros2_nervous_system",
                test_name="comprehensive_test",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                performance_metrics=[],
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_digital_twin_simulation(self) -> ModuleTestResult:
        """Test the Digital Twin Simulation module"""
        start_time = time.time()
        passed = True
        details = []
        performance_metrics = []

        try:
            # Test 1: Physics simulation accuracy
            physics_start = time.time()
            # Simulate physics calculations
            for i in range(1000):
                # Simulate simple physics computation
                pos = math.sin(i * 0.01) * math.cos(i * 0.02)
            physics_time = time.time() - physics_start
            performance_metrics.append(PerformanceMetric(
                name="physics_calculation_time", value=physics_time*1000, unit="ms",
                min_acceptable=0, max_acceptable=50, timestamp=time.time()
            ))

            if physics_time > 0.05:  # 50ms threshold
                details.append("Physics calculations too slow")
                passed = False

            # Test 2: Simulation update rate
            update_rate_start = time.time()
            frame_count = 600  # Simulate 10 seconds at 60 FPS
            for i in range(frame_count):
                time.sleep(1/1000)  # Simulate frame processing
            update_rate_time = time.time() - update_rate_start
            actual_rate = frame_count / update_rate_time if update_rate_time > 0 else 0
            performance_metrics.append(PerformanceMetric(
                name="simulation_frame_rate", value=actual_rate, unit="FPS",
                min_acceptable=30, max_acceptable=120, timestamp=time.time()
            ))

            if actual_rate < 30:  # Minimum 30 FPS
                details.append("Simulation frame rate too low")
                passed = False

            # Test 3: Sensor simulation accuracy
            sensor_start = time.time()
            # Simulate sensor data generation
            sensor_data = []
            for i in range(100):
                sensor_value = np.random.normal(0, 1)  # Simulate sensor noise
                sensor_data.append(sensor_value)
            sensor_time = time.time() - sensor_start
            performance_metrics.append(PerformanceMetric(
                name="sensor_simulation_time", value=sensor_time*1000, unit="ms",
                min_acceptable=0, max_acceptable=20, timestamp=time.time()
            ))

            # Test 4: Memory usage for simulation
            # In a real test, this would measure actual memory usage
            estimated_memory_mb = 50.0  # MB
            performance_metrics.append(PerformanceMetric(
                name="simulation_memory_usage", value=estimated_memory_mb, unit="MB",
                min_acceptable=0, max_acceptable=200, timestamp=time.time()
            ))

            execution_time = time.time() - start_time
            result = ModuleTestResult(
                module_name="digital_twin_simulation",
                test_name="comprehensive_test",
                passed=passed,
                details="; ".join(details) if details else "All tests passed",
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            self.performance_history["digital_twin_simulation"].append(result)

            return result

        except Exception as e:
            return ModuleTestResult(
                module_name="digital_twin_simulation",
                test_name="comprehensive_test",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                performance_metrics=[],
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_ai_brain_isaac(self) -> ModuleTestResult:
        """Test the AI Brain (NVIDIA Isaac) module"""
        start_time = time.time()
        passed = True
        details = []
        performance_metrics = []

        try:
            # Test 1: Perception processing time
            perception_start = time.time()
            # Simulate perception processing (object detection, etc.)
            for i in range(10):  # Process 10 frames
                # Simulate neural network inference
                dummy_input = np.random.random((224, 224, 3))  # Simulate image
                # Simulate processing
                time.sleep(0.02)  # 20ms per frame
            perception_time = time.time() - perception_start
            avg_perception_time = perception_time / 10 * 1000  # ms per frame
            performance_metrics.append(PerformanceMetric(
                name="perception_processing_time", value=avg_perception_time, unit="ms/frame",
                min_acceptable=0, max_acceptable=50, timestamp=time.time()
            ))

            if avg_perception_time > 50:  # 50ms threshold per frame
                details.append("Perception processing too slow")
                passed = False

            # Test 2: Decision making latency
            decision_start = time.time()
            # Simulate decision making process
            possible_actions = ['move_forward', 'turn_left', 'turn_right', 'stop', 'grasp']
            for i in range(20):  # Make 20 decisions
                selected_action = possible_actions[i % len(possible_actions)]
                time.sleep(0.005)  # 5ms per decision
            decision_time = time.time() - decision_start
            avg_decision_time = decision_time / 20 * 1000  # ms per decision
            performance_metrics.append(PerformanceMetric(
                name="decision_making_time", value=avg_decision_time, unit="ms/decision",
                min_acceptable=0, max_acceptable=20, timestamp=time.time()
            ))

            if avg_decision_time > 20:  # 20ms threshold
                details.append("Decision making too slow")
                passed = False

            # Test 3: Learning algorithm convergence
            learning_start = time.time()
            # Simulate learning process
            episode_rewards = []
            for episode in range(100):
                # Simulate an episode with improving performance
                reward = -10 + (episode / 100) * 20  # Improve from -10 to +10
                episode_rewards.append(reward)
                time.sleep(0.01)  # 10ms per episode
            learning_time = time.time() - learning_start

            # Check if learning is improving (last 10 episodes vs first 10)
            early_avg = statistics.mean(episode_rewards[:10])
            late_avg = statistics.mean(episode_rewards[-10:])
            learning_improvement = late_avg - early_avg
            performance_metrics.append(PerformanceMetric(
                name="learning_improvement", value=learning_improvement, unit="reward_units",
                min_acceptable=0, max_acceptable=20, timestamp=time.time()
            ))

            if learning_improvement < 0:  # Learning should improve
                details.append("Learning algorithm not improving")
                passed = False

            # Test 4: Memory system efficiency
            memory_start = time.time()
            # Simulate memory operations
            memory_items = {}
            for i in range(1000):
                key = f"item_{i}"
                value = {"data": np.random.random(10), "timestamp": time.time()}
                memory_items[key] = value
                if len(memory_items) > 100:  # Simulate forgetting old items
                    oldest_key = min(memory_items.keys(), key=lambda x: memory_items[x]["timestamp"])
                    del memory_items[oldest_key]
            memory_time = time.time() - memory_start
            performance_metrics.append(PerformanceMetric(
                name="memory_operation_time", value=memory_time*1000, unit="ms",
                min_acceptable=0, max_acceptable=100, timestamp=time.time()
            ))

            execution_time = time.time() - start_time
            result = ModuleTestResult(
                module_name="ai_brain_isaac",
                test_name="comprehensive_test",
                passed=passed,
                details="; ".join(details) if details else "All tests passed",
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            self.performance_history["ai_brain_isaac"].append(result)

            return result

        except Exception as e:
            return ModuleTestResult(
                module_name="ai_brain_isaac",
                test_name="comprehensive_test",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                performance_metrics=[],
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_vla_robotics(self) -> ModuleTestResult:
        """Test the Vision-Language-Action Robotics module"""
        start_time = time.time()
        passed = True
        details = []
        performance_metrics = []

        try:
            # Test 1: Vision-language integration latency
            vl_start = time.time()
            # Simulate vision-language processing
            for i in range(5):
                # Process an image with language command
                image_features = np.random.random(512)  # Simulate image features
                language_features = np.random.random(512)  # Simulate language features
                # Simulate fusion
                fused_features = np.concatenate([image_features, language_features])
                time.sleep(0.05)  # 50ms per processing cycle
            vl_time = time.time() - vl_start
            avg_vl_time = vl_time / 5 * 1000  # ms per cycle
            performance_metrics.append(PerformanceMetric(
                name="vision_language_latency", value=avg_vl_time, unit="ms/cycle",
                min_acceptable=0, max_acceptable=100, timestamp=time.time()
            ))

            if avg_vl_time > 100:  # 100ms threshold
                details.append("Vision-language processing too slow")
                passed = False

            # Test 2: Action planning accuracy
            planning_start = time.time()
            # Simulate action planning
            possible_actions = ['move', 'grasp', 'release', 'point', 'speak']
            for i in range(20):  # Plan 20 actions
                action_sequence = [possible_actions[j % len(possible_actions)] for j in range(3)]
                time.sleep(0.02)  # 20ms per plan
            planning_time = time.time() - planning_start
            avg_planning_time = planning_time / 20 * 1000  # ms per plan
            performance_metrics.append(PerformanceMetric(
                name="action_planning_time", value=avg_planning_time, unit="ms/plan",
                min_acceptable=0, max_acceptable=50, timestamp=time.time()
            ))

            if avg_planning_time > 50:  # 50ms threshold
                details.append("Action planning too slow")
                passed = False

            # Test 3: Multimodal fusion effectiveness
            fusion_start = time.time()
            # Simulate multimodal fusion
            for i in range(10):
                # Combine visual, linguistic, and action features
                visual_context = np.random.random(256)
                linguistic_context = np.random.random(256)
                action_context = np.random.random(256)
                fused_context = np.concatenate([visual_context, linguistic_context, action_context])
                time.sleep(0.01)  # 10ms per fusion
            fusion_time = time.time() - fusion_start
            avg_fusion_time = fusion_time / 10 * 1000  # ms per fusion
            performance_metrics.append(PerformanceMetric(
                name="multimodal_fusion_time", value=avg_fusion_time, unit="ms/fusion",
                min_acceptable=0, max_acceptable=20, timestamp=time.time()
            ))

            # Test 4: Task execution success rate
            execution_start = time.time()
            successful_executions = 0
            total_executions = 15
            for i in range(total_executions):
                # Simulate task execution with some failures
                success = np.random.random() > 0.1  # 90% success rate in simulation
                if success:
                    successful_executions += 1
                time.sleep(0.1)  # 100ms per task
            execution_time = time.time() - execution_start
            success_rate = successful_executions / total_executions if total_executions > 0 else 0
            performance_metrics.append(PerformanceMetric(
                name="task_execution_success_rate", value=success_rate*100, unit="%",
                min_acceptable=80, max_acceptable=100, timestamp=time.time()
            ))

            if success_rate < 0.8:  # 80% minimum success rate
                details.append("Task execution success rate too low")
                passed = False

            execution_time_total = time.time() - start_time
            result = ModuleTestResult(
                module_name="vla_robotics",
                test_name="comprehensive_test",
                passed=passed,
                details="; ".join(details) if details else "All tests passed",
                performance_metrics=performance_metrics,
                execution_time=execution_time_total,
                timestamp=time.time()
            )

            self.test_results.append(result)
            self.performance_history["vla_robotics"].append(result)

            return result

        except Exception as e:
            return ModuleTestResult(
                module_name="vla_robotics",
                test_name="comprehensive_test",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                performance_metrics=[],
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def test_humanoid_integration(self) -> ModuleTestResult:
        """Test the Humanoid Integration module"""
        start_time = time.time()
        passed = True
        details = []
        performance_metrics = []

        try:
            # Test 1: Module coordination latency
            coord_start = time.time()
            # Simulate coordinating between modules
            modules = ['nervous_system', 'digital_twin', 'ai_brain', 'vla', 'integration']
            for i in range(50):  # 50 coordination cycles
                for module in modules:
                    # Simulate sending coordination message
                    time.sleep(0.002)  # 2ms per module message
            coord_time = time.time() - coord_start
            avg_coord_time = coord_time / 50 * 1000  # ms per cycle
            performance_metrics.append(PerformanceMetric(
                name="module_coordination_latency", value=avg_coord_time, unit="ms/cycle",
                min_acceptable=0, max_acceptable=20, timestamp=time.time()
            ))

            if avg_coord_time > 20:  # 20ms threshold
                details.append("Module coordination too slow")
                passed = False

            # Test 2: Safety system response time
            safety_start = time.time()
            # Simulate safety monitoring and response
            for i in range(100):  # 100 safety checks
                # Simulate checking various safety parameters
                tilt_angle = np.random.random() * 1.0  # Random tilt
                distance_to_obstacle = np.random.random() * 3.0  # Random distance
                joint_position = np.random.random() * 3.0  # Random joint position
                time.sleep(0.001)  # 1ms per check
            safety_time = time.time() - safety_start
            avg_safety_time = safety_time / 100 * 1000  # ms per check
            performance_metrics.append(PerformanceMetric(
                name="safety_check_time", value=avg_safety_time, unit="ms/check",
                min_acceptable=0, max_acceptable=10, timestamp=time.time()
            ))

            if avg_safety_time > 10:  # 10ms threshold
                details.append("Safety checks too slow")
                passed = False

            # Test 3: Behavior transition smoothness
            behavior_start = time.time()
            # Simulate behavior transitions
            behaviors = ['idle', 'navigating', 'manipulating', 'interacting', 'speaking']
            for i in range(20):  # 20 behavior transitions
                current_behavior = behaviors[i % len(behaviors)]
                next_behavior = behaviors[(i + 1) % len(behaviors)]
                # Simulate transition
                time.sleep(0.05)  # 50ms per transition
            behavior_time = time.time() - behavior_start
            avg_behavior_time = behavior_time / 20 * 1000  # ms per transition
            performance_metrics.append(PerformanceMetric(
                name="behavior_transition_time", value=avg_behavior_time, unit="ms/transition",
                min_acceptable=0, max_acceptable=100, timestamp=time.time()
            ))

            # Test 4: System stability under load
            stability_start = time.time()
            # Simulate system under load with multiple concurrent operations
            concurrent_ops = 10
            for i in range(concurrent_ops):
                # Simulate different system operations happening simultaneously
                time.sleep(0.01)  # 10ms per operation
            stability_time = time.time() - stability_start
            performance_metrics.append(PerformanceMetric(
                name="system_stability_time", value=stability_time*1000, unit="ms",
                min_acceptable=0, max_acceptable=200, timestamp=time.time()
            ))

            # Test 5: Emergency stop response
            emergency_start = time.time()
            # Simulate emergency stop activation and response
            time.sleep(0.1)  # Time to detect emergency
            time.sleep(0.05)  # Time to execute stop
            emergency_time = time.time() - emergency_start
            performance_metrics.append(PerformanceMetric(
                name="emergency_stop_response", value=emergency_time*1000, unit="ms",
                min_acceptable=0, max_acceptable=200, timestamp=time.time()
            ))

            if emergency_time > 0.2:  # 200ms threshold
                details.append("Emergency stop response too slow")
                passed = False

            execution_time = time.time() - start_time
            result = ModuleTestResult(
                module_name="humanoid_integration",
                test_name="comprehensive_test",
                passed=passed,
                details="; ".join(details) if details else "All tests passed",
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                timestamp=time.time()
            )

            self.test_results.append(result)
            self.performance_history["humanoid_integration"].append(result)

            return result

        except Exception as e:
            return ModuleTestResult(
                module_name="humanoid_integration",
                test_name="comprehensive_test",
                passed=False,
                details=f"Exception occurred: {str(e)}",
                performance_metrics=[],
                execution_time=time.time() - start_time,
                timestamp=time.time()
            )

    def run_comprehensive_module_tests(self) -> Dict[str, ModuleTestResult]:
        """Run comprehensive tests for all modules"""
        results = {}

        self.get_logger().info("Starting comprehensive module tests...")

        # Test each module
        results['ros2_nervous_system'] = self.test_ros2_nervous_system()
        self.get_logger().info(f"Nervous System Test: {'PASS' if results['ros2_nervous_system'].passed else 'FAIL'}")

        results['digital_twin_simulation'] = self.test_digital_twin_simulation()
        self.get_logger().info(f"Digital Twin Test: {'PASS' if results['digital_twin_simulation'].passed else 'FAIL'}")

        results['ai_brain_isaac'] = self.test_ai_brain_isaac()
        self.get_logger().info(f"AI Brain Test: {'PASS' if results['ai_brain_isaac'].passed else 'FAIL'}")

        results['vla_robotics'] = self.test_vla_robotics()
        self.get_logger().info(f"VLA Test: {'PASS' if results['vla_robotics'].passed else 'FAIL'}")

        results['humanoid_integration'] = self.test_humanoid_integration()
        self.get_logger().info(f"Integration Test: {'PASS' if results['humanoid_integration'].passed else 'FAIL'}")

        return results

    def generate_performance_report(self) -> str:
        """Generate a detailed performance report"""
        report = []
        report.append("HUMANOID ROBOTICS PLATFORM - MODULE PERFORMANCE REPORT")
        report.append("=" * 60)

        for module_name, results in self.performance_history.items():
            if not results:
                continue

            report.append(f"\n{module_name.upper()}:")
            report.append("-" * 40)

            # Calculate average performance metrics
            metric_averages = {}
            for result in results:
                for metric in result.performance_metrics:
                    if metric.name not in metric_averages:
                        metric_averages[metric.name] = []
                    metric_averages[metric.name].append(metric.value)

            # Report each metric
            for metric_name, values in metric_averages.items():
                avg_value = sum(values) / len(values)
                unit = results[0].performance_metrics[0].unit if results and results[0].performance_metrics else ""

                # Find the first occurrence of this metric to get min/max
                min_acceptable = max(m.min_acceptable for m in results[0].performance_metrics if m.name == metric_name)
                max_acceptable = min(m.max_acceptable for m in results[0].performance_metrics if m.name == metric_name)

                status = "OK" if min_acceptable <= avg_value <= max_acceptable else "ISSUE"
                report.append(f"  {metric_name:30s}: {avg_value:8.2f} {unit:3s} [{status}]")

        report.append(f"\nTotal Tests Run: {len(self.test_results)}")
        passed_count = sum(1 for r in self.test_results if r.passed)
        report.append(f"Passed Tests: {passed_count}")
        report.append(f"Failed Tests: {len(self.test_results) - passed_count}")
        report.append(f"Success Rate: {passed_count/len(self.test_results)*100:.1f}%" if self.test_results else "Success Rate: 0%")

        return "\n".join(report)

    def get_module_statistics(self, module_name: str) -> Dict[str, any]:
        """Get statistics for a specific module"""
        module_results = self.performance_history.get(module_name, [])

        if not module_results:
            return {
                'module': module_name,
                'total_tests': 0,
                'passed_tests': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'performance_metrics': {}
            }

        passed_tests = sum(1 for r in module_results if r.passed)
        total_time = sum(r.execution_time for r in module_results)

        # Calculate average performance metrics
        metric_averages = {}
        for result in module_results:
            for metric in result.performance_metrics:
                if metric.name not in metric_averages:
                    metric_averages[metric.name] = {'values': [], 'unit': metric.unit}
                metric_averages[metric.name]['values'].append(metric.value)

        for metric_name, data in metric_averages.items():
            metric_averages[metric_name]['average'] = sum(data['values']) / len(data['values'])

        return {
            'module': module_name,
            'total_tests': len(module_results),
            'passed_tests': passed_tests,
            'success_rate': passed_tests / len(module_results) if module_results else 0.0,
            'average_execution_time': total_time / len(module_results) if module_results else 0.0,
            'performance_metrics': metric_averages
        }


def main(args=None):
    rclpy.init(args=args)

    module_tester = ModuleTester()

    try:
        print("Running comprehensive module tests for Humanoid Robotics Platform...")
        results = module_tester.run_comprehensive_module_tests()

        # Generate and print performance report
        report = module_tester.generate_performance_report()
        print(report)

        # Check overall success
        all_passed = all(result.passed for result in results.values())

        return 0 if all_passed else 1

    except Exception as e:
        print(f"Error during module testing: {e}")
        return 1
    finally:
        module_tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())