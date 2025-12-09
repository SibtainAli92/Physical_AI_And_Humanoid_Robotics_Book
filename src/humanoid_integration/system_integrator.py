#!/usr/bin/env python3
# system_integrator.py
# System integration framework for all humanoid robot modules

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String, Bool, Float64
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from humanoid_msgs.msg import HumanoidControlCommand, HumanoidSensorData
from builtin_interfaces.msg import Time
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import subprocess
import psutil
import json


@dataclass
class ModuleStatus:
    """Data class for module status"""
    name: str
    status: str  # 'running', 'stopped', 'error', 'initializing'
    cpu_usage: float
    memory_usage: float
    last_heartbeat: Time
    topics: List[str]


class SystemIntegrator(Node):
    """
    System integration framework for all humanoid robot modules
    """
    def __init__(self):
        super().__init__('system_integrator')

        # Declare parameters
        self.declare_parameter('heartbeat_interval', 1.0)  # seconds
        self.declare_parameter('module_monitoring_interval', 2.0)  # seconds
        self.declare_parameter('system_status_publish_rate', 5.0)  # Hz
        self.declare_parameter('critical_modules', [
            'ros2_nervous_system',
            'safety_monitor',
            'controller_manager'
        ])

        self.heartbeat_interval = self.get_parameter('heartbeat_interval').value
        self.module_monitoring_interval = self.get_parameter('module_monitoring_interval').value
        self.system_status_publish_rate = self.get_parameter('system_status_publish_rate').value
        self.critical_modules = self.get_parameter('critical_modules').value

        # Initialize module tracking
        self.modules: Dict[str, ModuleStatus] = {}
        self.module_processes: Dict[str, subprocess.Popen] = {}
        self.module_heartbeats: Dict[str, Time] = {}

        # Publishers
        self.system_status_pub = self.create_publisher(
            String,
            '/system/status',
            QoSProfile(depth=10)
        )

        self.system_health_pub = self.create_publisher(
            String,
            '/system/health',
            QoSProfile(depth=10)
        )

        self.heartbeat_pub = self.create_publisher(
            String,
            '/system/heartbeat',
            QoSProfile(depth=10)
        )

        # Subscribers
        self.heartbeat_sub = self.create_subscription(
            String,
            '/system/heartbeat',
            self.heartbeat_callback,
            QoSProfile(depth=10)
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10)
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            QoSProfile(depth=10)
        )

        # Create timers
        self.heartbeat_timer = self.create_timer(
            self.heartbeat_interval,
            self.publish_heartbeat
        )

        self.monitoring_timer = self.create_timer(
            self.module_monitoring_interval,
            self.monitor_modules
        )

        self.status_timer = self.create_timer(
            1.0/self.system_status_publish_rate,
            self.publish_system_status
        )

        # Initialize system
        self.initialize_modules()
        self.system_initialized = True

        self.get_logger().info('System Integrator initialized')

    def initialize_modules(self):
        """
        Initialize all system modules
        """
        # Register core modules
        self.register_module('ros2_nervous_system')
        self.register_module('digital_twin_simulation')
        self.register_module('ai_brain_isaac')
        self.register_module('vla_robotics')
        self.register_module('safety_system')
        self.register_module('controller_manager')

        self.get_logger().info('All modules registered')

    def register_module(self, module_name: str):
        """
        Register a module with the system integrator
        """
        self.modules[module_name] = ModuleStatus(
            name=module_name,
            status='initializing',
            cpu_usage=0.0,
            memory_usage=0.0,
            last_heartbeat=self.get_clock().now().to_msg(),
            topics=[]
        )
        self.get_logger().info(f'Registered module: {module_name}')

    def heartbeat_callback(self, msg: String):
        """
        Callback for heartbeat messages from modules
        """
        module_name = msg.data.split(':')[0] if ':' in msg.data else msg.data

        if module_name in self.modules:
            self.modules[module_name].last_heartbeat = self.get_clock().now().to_msg()
            self.modules[module_name].status = 'running'

    def joint_state_callback(self, msg: JointState):
        """
        Callback for joint state messages (integration point)
        """
        # This is where we can integrate joint state data across modules
        pass

    def imu_callback(self, msg: Imu):
        """
        Callback for IMU messages (integration point)
        """
        # This is where we can integrate IMU data across modules
        pass

    def publish_heartbeat(self):
        """
        Publish system heartbeat
        """
        heartbeat_msg = String()
        heartbeat_msg.data = f"system_integrator:{self.get_clock().now().nanoseconds}"
        self.heartbeat_pub.publish(heartbeat_msg)

    def monitor_modules(self):
        """
        Monitor the status of all registered modules
        """
        current_time = self.get_clock().now()

        for module_name, module_status in self.modules.items():
            # Update resource usage
            module_status.cpu_usage = psutil.cpu_percent()
            module_status.memory_usage = psutil.virtual_memory().percent

            # Check if module is responsive
            time_since_heartbeat = (
                current_time - rclpy.time.Time.from_msg(module_status.last_heartbeat)
            ).nanoseconds / 1e9

            if time_since_heartbeat > self.heartbeat_interval * 3:
                module_status.status = 'unresponsive'
                self.get_logger().warn(f'Module {module_name} is unresponsive')

                # Restart critical modules if they become unresponsive
                if module_name in self.critical_modules:
                    self.restart_module(module_name)
            else:
                module_status.status = 'running'

    def restart_module(self, module_name: str):
        """
        Restart a specific module
        """
        self.get_logger().info(f'Restarting module: {module_name}')

        # In a real implementation, this would restart the actual process
        # For simulation, we'll just reset the status
        if module_name in self.modules:
            self.modules[module_name].status = 'restarting'
            time.sleep(0.5)  # Simulate restart time
            self.modules[module_name].status = 'running'
            self.modules[module_name].last_heartbeat = self.get_clock().now().to_msg()

    def publish_system_status(self):
        """
        Publish overall system status
        """
        # Create status summary
        status_summary = {
            'timestamp': self.get_clock().now().nanoseconds,
            'modules': {},
            'overall_status': 'operational',
            'critical_issues': []
        }

        for module_name, module_status in self.modules.items():
            status_summary['modules'][module_name] = {
                'status': module_status.status,
                'cpu_usage': module_status.cpu_usage,
                'memory_usage': module_status.memory_usage
            }

            if module_status.status == 'error' or module_status.status == 'unresponsive':
                status_summary['critical_issues'].append(module_name)

        # Determine overall system status
        if status_summary['critical_issues']:
            status_summary['overall_status'] = 'degraded'
            if any(mod in self.critical_modules for mod in status_summary['critical_issues']):
                status_summary['overall_status'] = 'critical'

        # Publish system status
        status_msg = String()
        status_msg.data = json.dumps(status_summary)
        self.system_status_pub.publish(status_msg)

        # Also publish health status
        health_msg = String()
        health_msg.data = status_summary['overall_status']
        self.system_health_pub.publish(health_msg)

    def get_module_status(self, module_name: str) -> Optional[ModuleStatus]:
        """
        Get status of a specific module
        """
        return self.modules.get(module_name)

    def get_system_health(self) -> str:
        """
        Get overall system health status
        """
        operational_count = sum(1 for status in self.modules.values()
                               if status.status in ['running', 'initializing'])
        total_count = len(self.modules)

        if total_count == 0:
            return 'unknown'

        operational_ratio = operational_count / total_count

        if operational_ratio == 1.0:
            return 'healthy'
        elif operational_ratio >= 0.8:
            return 'degraded'
        else:
            return 'critical'

    def start_module(self, module_name: str):
        """
        Start a specific module
        """
        if module_name in self.modules:
            # In a real implementation, this would start the actual process
            self.modules[module_name].status = 'running'
            self.modules[module_name].last_heartbeat = self.get_clock().now().to_msg()
            self.get_logger().info(f'Started module: {module_name}')

    def stop_module(self, module_name: str):
        """
        Stop a specific module
        """
        if module_name in self.modules:
            # In a real implementation, this would stop the actual process
            self.modules[module_name].status = 'stopped'
            self.get_logger().info(f'Stopped module: {module_name}')

    def get_data_flow_map(self) -> Dict[str, List[str]]:
        """
        Get the data flow map between modules
        """
        # Define data flow between modules
        data_flow = {
            'ros2_nervous_system': [
                'joint_states', 'imu_data', 'control_commands'
            ],
            'digital_twin_simulation': [
                'simulation_data', 'ground_truth', 'sensor_simulation'
            ],
            'ai_brain_isaac': [
                'perception_data', 'navigation_goals', 'behavior_commands'
            ],
            'vla_robotics': [
                'nlp_commands', 'speech_output', 'gesture_commands'
            ],
            'safety_system': [
                'safety_status', 'emergency_stop', 'limit_violations'
            ]
        }
        return data_flow


class DataFlowManager(Node):
    """
    Data flow management between modules
    """
    def __init__(self):
        super().__init__('data_flow_manager')

        # Declare parameters
        self.declare_parameter('max_buffer_size', 100)
        self.declare_parameter('buffer_cleanup_interval', 5.0)  # seconds

        self.max_buffer_size = self.get_parameter('max_buffer_size').value
        self.buffer_cleanup_interval = self.get_parameter('buffer_cleanup_interval').value

        # Initialize data buffers
        self.data_buffers = {}
        self.buffer_locks = {}

        # Create cleanup timer
        self.cleanup_timer = self.create_timer(
            self.buffer_cleanup_interval,
            self.cleanup_buffers
        )

        self.get_logger().info('Data Flow Manager initialized')

    def register_data_stream(self, stream_name: str):
        """
        Register a data stream for buffering
        """
        if stream_name not in self.data_buffers:
            self.data_buffers[stream_name] = []
            self.buffer_locks[stream_name] = threading.Lock()

    def publish_to_stream(self, stream_name: str, data):
        """
        Publish data to a specific stream
        """
        if stream_name not in self.data_buffers:
            self.register_data_stream(stream_name)

        with self.buffer_locks[stream_name]:
            self.data_buffers[stream_name].append(data)

            # Keep buffer size within limits
            if len(self.data_buffers[stream_name]) > self.max_buffer_size:
                self.data_buffers[stream_name] = self.data_buffers[stream_name][-self.max_buffer_size:]

    def subscribe_to_stream(self, stream_name: str, callback: Callable, count: int = 1):
        """
        Subscribe to a data stream
        """
        if stream_name not in self.data_buffers:
            return []

        with self.buffer_locks[stream_name]:
            if count == -1:  # Get all available data
                data = self.data_buffers[stream_name][:]
            else:
                data = self.data_buffers[stream_name][-count:]

        # Call the callback with the data
        if callback and data:
            for item in data:
                callback(item)

        return data

    def cleanup_buffers(self):
        """
        Periodically clean up old buffer data
        """
        # This is where we might implement more sophisticated buffer management
        # For now, just log the buffer sizes
        for stream_name, buffer in self.data_buffers.items():
            self.get_logger().debug(f'Buffer {stream_name}: {len(buffer)} items')


class SystemMonitor(Node):
    """
    System monitoring and diagnostics
    """
    def __init__(self):
        super().__init__('system_monitor')

        # Publishers
        self.diagnostics_pub = self.create_publisher(
            String,
            '/system/diagnostics',
            QoSProfile(depth=10)
        )

        # Initialize monitoring
        self.resource_monitor_timer = self.create_timer(1.0, self.monitor_resources)
        self.diagnostic_timer = self.create_timer(5.0, self.publish_diagnostics)

        self.get_logger().info('System Monitor initialized')

    def monitor_resources(self):
        """
        Monitor system resources
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        # Log warnings if resources are high
        if cpu_percent > 80:
            self.get_logger().warn(f'High CPU usage: {cpu_percent}%')
        if memory_percent > 80:
            self.get_logger().warn(f'High memory usage: {memory_percent}%')

    def publish_diagnostics(self):
        """
        Publish system diagnostics
        """
        diagnostics = {
            'timestamp': self.get_clock().now().nanoseconds,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'temperature': self.get_system_temperature(),
            'network_io': self.get_network_io(),
            'process_count': len(psutil.pids())
        }

        diag_msg = String()
        diag_msg.data = json.dumps(diagnostics)
        self.diagnostics_pub.publish(diag_msg)

    def get_system_temperature(self) -> float:
        """
        Get system temperature (simulated)
        """
        # In a real system, this would read from hardware sensors
        # For simulation, return a reasonable value
        return 45.0

    def get_network_io(self) -> Dict:
        """
        Get network I/O statistics
        """
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }


def main(args=None):
    rclpy.init(args=args)

    # Create integration nodes
    system_integrator = SystemIntegrator()
    data_flow_manager = DataFlowManager()
    system_monitor = SystemMonitor()

    # Use a MultiThreadedExecutor to handle callbacks from multiple nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(system_integrator)
    executor.add_node(data_flow_manager)
    executor.add_node(system_monitor)

    try:
        executor.spin()
    except KeyboardInterrupt:
        system_integrator.get_logger().info('System integrator interrupted by user')
        data_flow_manager.get_logger().info('Data flow manager interrupted by user')
        system_monitor.get_logger().info('System monitor interrupted by user')
    finally:
        system_integrator.destroy_node()
        data_flow_manager.destroy_node()
        system_monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()