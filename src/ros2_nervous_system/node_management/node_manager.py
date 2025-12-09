#!/usr/bin/env python3
# node_manager.py
# Node lifecycle management and monitoring for humanoid robot

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from lifecycle_msgs.msg import Transition, TransitionEvent
from lifecycle_msgs.srv import ChangeState, GetState, GetAvailableStates, GetAvailableTransitions
from std_msgs.msg import String, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import time
import threading
from typing import Dict, List, Optional
import subprocess
import psutil
from enum import Enum


class NodeState(Enum):
    """Enumeration of possible node states for monitoring"""
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    ERROR = "error"
    UNKNOWN = "unknown"


class NodeManager(LifecycleNode):
    """
    Node lifecycle management system for humanoid robot
    """
    def __init__(self):
        super().__init__('node_manager')

        # Declare parameters
        self.declare_parameter('monitoring_interval', 1.0)  # seconds
        self.declare_parameter('restart_attempts', 3)
        self.declare_parameter('restart_delay', 2.0)  # seconds
        self.declare_parameter('critical_nodes', [
            'robot_state_publisher',
            'controller_manager',
            'safety_monitor'
        ])

        self.monitoring_interval = self.get_parameter('monitoring_interval').value
        self.restart_attempts = self.get_parameter('restart_attempts').value
        self.restart_delay = self.get_parameter('restart_delay').value
        self.critical_nodes = self.get_parameter('critical_nodes').value

        # Node monitoring data
        self.monitored_nodes = {}  # node_name -> {state, last_seen, restart_count}
        self.node_processes = {}  # node_name -> process info

        # Publishers
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.status_pub = self.create_publisher(String, '/node_manager/status', 10)

        # Timers
        self.monitoring_timer = self.create_timer(
            self.monitoring_interval,
            self.monitor_nodes
        )

        # Initialize
        self.get_logger().info('Node Manager initialized')

    def monitor_nodes(self):
        """
        Monitor the status of managed nodes
        """
        # This would typically query the ROS2 graph to see what nodes are running
        # For now, we'll simulate monitoring of some key nodes
        active_nodes = self.get_node_names()

        # Update diagnostics
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        for node_name in active_nodes:
            if node_name in self.critical_nodes:
                status = DiagnosticStatus()
                status.name = f"Node: {node_name}"
                status.level = DiagnosticStatus.OK
                status.message = "Node is running normally"
                status.hardware_id = node_name
                diag_array.status.append(status)

        self.diag_pub.publish(diag_array)

        # Check for nodes that should be running but aren't
        self.check_critical_nodes()

    def check_critical_nodes(self):
        """
        Check if critical nodes are running and restart if necessary
        """
        active_nodes = set(self.get_node_names())

        for node_name in self.critical_nodes:
            if node_name not in active_nodes:
                self.get_logger().warn(f'Critical node {node_name} is not running')
                self.restart_node(node_name)

    def restart_node(self, node_name: str):
        """
        Attempt to restart a failed node
        """
        if node_name not in self.node_processes:
            self.get_logger().info(f'No process information for {node_name}, cannot restart')
            return

        # Get restart count for this node
        if node_name not in self.monitored_nodes:
            self.monitored_nodes[node_name] = {
                'restart_count': 0,
                'last_restart': 0
            }

        restart_info = self.monitored_nodes[node_name]
        restart_info['restart_count'] += 1

        if restart_info['restart_count'] > self.restart_attempts:
            self.get_logger().error(f'Max restart attempts reached for {node_name}')
            # Send alert about node failure
            status_msg = String()
            status_msg.data = f"CRITICAL: Node {node_name} failed to restart after {self.restart_attempts} attempts"
            self.status_pub.publish(status_msg)
            return

        self.get_logger().info(f'Attempting to restart {node_name} (attempt {restart_info["restart_count"]})')

        # Restart the process (this is a simplified example)
        # In practice, you'd need to know how each node was started
        try:
            # For this example, we'll just log that we're attempting restart
            # In a real system, this might involve launching a launch file
            # or restarting a specific process
            self.get_logger().info(f'Restarted node {node_name}')
            restart_info['last_restart'] = time.time()

            # Reset restart count after successful restart
            time.sleep(self.restart_delay)  # Wait before resetting

        except Exception as e:
            self.get_logger().error(f'Failed to restart {node_name}: {str(e)}')

    def add_node_for_monitoring(self, node_name: str, process_info: Dict = None):
        """
        Add a node to be monitored
        """
        self.monitored_nodes[node_name] = {
            'state': NodeState.UNKNOWN,
            'last_seen': time.time(),
            'restart_count': 0,
            'process_info': process_info
        }
        self.get_logger().info(f'Added {node_name} to monitoring')

    def get_system_resources(self):
        """
        Get system resource information
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent
        }

    def get_node_states(self):
        """
        Get current states of all monitored nodes
        """
        states = {}
        for node_name in self.monitored_nodes:
            # In a real implementation, this would query the actual node state
            states[node_name] = self.monitored_nodes[node_name]['state']
        return states


class NodeMonitorService(Node):
    """
    Service interface for node monitoring
    """
    def __init__(self):
        super().__init__('node_monitor_service')

        # Service servers
        self.get_state_srv = self.create_service(
            GetState,
            'node_monitor/get_state',
            self.get_state_callback
        )

        self.change_state_srv = self.create_service(
            ChangeState,
            'node_monitor/change_state',
            self.change_state_callback
        )

        self.get_available_states_srv = self.create_service(
            GetAvailableStates,
            'node_monitor/get_available_states',
            self.get_available_states_callback
        )

        self.get_available_transitions_srv = self.create_service(
            GetAvailableTransitions,
            'node_monitor/get_available_transitions',
            self.get_available_transitions_callback
        )

        self.get_logger().info('Node Monitor Service initialized')

    def get_state_callback(self, request, response):
        """
        Get the current state of a node
        """
        # In a real implementation, this would query the actual node
        response.current_state.label = "active"  # Default state
        response.current_state.id = Transition.TRANSITION_CONFIGURE
        return response

    def change_state_callback(self, request, response):
        """
        Request a state change for a node
        """
        # In a real implementation, this would command the actual node
        response.success = True
        return response

    def get_available_states_callback(self, request, response):
        """
        Get available states for a node
        """
        # Add available states
        state1 = TransitionEvent()
        state1.transition.id = Transition.TRANSITION_CONFIGURE
        state1.transition.label = "unconfigured"
        response.available_states.append(state1)

        state2 = TransitionEvent()
        state2.transition.id = Transition.TRANSITION_ACTIVATE
        state2.transition.label = "active"
        response.available_states.append(state2)

        return response

    def get_available_transitions_callback(self, request, response):
        """
        Get available transitions for a node
        """
        # Add available transitions
        trans1 = TransitionEvent()
        trans1.transition.id = Transition.TRANSITION_CONFIGURE
        trans1.transition.label = "configure"
        response.available_transitions.append(trans1)

        trans2 = TransitionEvent()
        trans2.transition.id = Transition.TRANSITION_ACTIVATE
        trans2.transition.label = "activate"
        response.available_transitions.append(trans2)

        return response


def main(args=None):
    rclpy.init(args=args)

    node_manager = NodeManager()
    node_monitor_service = NodeMonitorService()

    # Add some critical nodes for monitoring
    node_manager.add_node_for_monitoring('robot_state_publisher')
    node_manager.add_node_for_monitoring('controller_manager')
    node_manager.add_node_for_monitoring('safety_monitor')

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node_manager)
    executor.add_node(node_monitor_service)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node_manager.get_logger().info('Node Manager interrupted by user')
    finally:
        node_manager.destroy_node()
        node_monitor_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()