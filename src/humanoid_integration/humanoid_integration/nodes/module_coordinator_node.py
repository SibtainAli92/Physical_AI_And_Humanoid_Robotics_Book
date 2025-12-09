#!/usr/bin/env python3

"""
Module Coordinator Node for Humanoid Robotics Platform

This node coordinates the operation of different system modules,
managing their startup, shutdown, and interaction patterns.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import subprocess
import signal
import os


class ModuleName(Enum):
    NERVOUS_SYSTEM = "ros2_nervous_system"
    DIGITAL_TWIN = "digital_twin_simulation"
    AI_BRAIN = "ai_brain_isaac"
    VLA_ROBOTICS = "vla_robotics"
    INTEGRATION = "humanoid_integration"


class ModuleState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ModuleInfo:
    """Information about a system module"""
    name: ModuleName
    state: ModuleState
    pid: Optional[int]
    launch_file: str
    dependencies: List[ModuleName]
    health_score: float
    last_update: float


class ModuleCoordinatorNode(Node):
    def __init__(self):
        super().__init__('module_coordinator_node')

        # Publishers for coordination
        self.coordinator_status_publisher = self.create_publisher(String, 'coordinator/status', 10)
        self.module_control_publisher = self.create_publisher(String, 'coordinator/module_control', 10)
        self.system_health_publisher = self.create_publisher(Float32, 'coordinator/system_health', 10)

        # Subscribers for module status
        self.nervous_system_status_subscriber = self.create_subscription(
            String, 'nervous_system/status', self.module_status_callback, 10)
        self.digital_twin_status_subscriber = self.create_subscription(
            String, 'digital_twin/status', self.module_status_callback, 10)
        self.ai_brain_status_subscriber = self.create_subscription(
            String, 'ai_brain/status', self.module_status_callback, 10)
        self.vla_status_subscriber = self.create_subscription(
            String, 'vla/status', self.module_status_callback, 10)
        self.integration_status_subscriber = self.create_subscription(
            String, 'integration/status', self.module_status_callback, 10)

        # Subscribers for coordination requests
        self.coordinator_command_subscriber = self.create_subscription(
            String, 'coordinator/command', self.coordinator_command_callback, 10)

        # Timer for module management
        self.coordinator_timer = self.create_timer(1.0, self.coordinator_callback)  # 1 Hz

        # Module management state
        self.modules: Dict[ModuleName, ModuleInfo] = {}
        self.coordinator_enabled = True
        self.system_initialized = False
        self.sim_time = time.time()
        self.last_coordinator_time = time.time()

        # Initialize module information
        self.initialize_modules()

        # Module startup sequence
        self.startup_sequence = [
            ModuleName.NERVOUS_SYSTEM,
            ModuleName.DIGITAL_TWIN,
            ModuleName.AI_BRAIN,
            ModuleName.VLA_ROBOTICS,
            ModuleName.INTEGRATION
        ]

        # Module shutdown sequence (reverse of startup)
        self.shutdown_sequence = list(reversed(self.startup_sequence))

        # Threading lock for module management
        self.module_lock = threading.Lock()

        self.get_logger().info('Module Coordinator Node initialized')

    def initialize_modules(self):
        """Initialize module information"""
        module_configs = {
            ModuleName.NERVOUS_SYSTEM: {
                'launch_file': 'central_nervous_node',
                'dependencies': [],
            },
            ModuleName.DIGITAL_TWIN: {
                'launch_file': 'simulation_environment_node',
                'dependencies': [ModuleName.NERVOUS_SYSTEM],
            },
            ModuleName.AI_BRAIN: {
                'launch_file': 'cognitive_controller_node',
                'dependencies': [ModuleName.NERVOUS_SYSTEM],
            },
            ModuleName.VLA_ROBOTICS: {
                'launch_file': 'task_execution_node',
                'dependencies': [ModuleName.AI_BRAIN, ModuleName.NERVOUS_SYSTEM],
            },
            ModuleName.INTEGRATION: {
                'launch_file': 'system_integration_node',
                'dependencies': [ModuleName.NERVOUS_SYSTEM, ModuleName.DIGITAL_TWIN, ModuleName.AI_BRAIN, ModuleName.VLA_ROBOTICS],
            }
        }

        for module_name, config in module_configs.items():
            self.modules[module_name] = ModuleInfo(
                name=module_name,
                state=ModuleState.STOPPED,
                pid=None,
                launch_file=config['launch_file'],
                dependencies=config['dependencies'],
                health_score=0.0,
                last_update=0.0
            )

    def module_status_callback(self, msg):
        """Callback for module status updates"""
        # Extract module name from topic or message
        # This is a simplified approach - in reality, you'd identify the source module
        # For now, we'll just update the health score based on the message content
        with self.module_lock:
            # Determine which module sent the status based on message content
            module_name = self.identify_module_from_status(msg.data)
            if module_name and module_name in self.modules:
                # Update health score based on status message
                health = self.estimate_health_from_status(msg.data)
                self.modules[module_name].health_score = health
                self.modules[module_name].last_update = time.time()

    def identify_module_from_status(self, status_msg: str) -> Optional[ModuleName]:
        """Identify which module sent a status message"""
        status_lower = status_msg.lower()

        if 'nervous' in status_lower or 'central' in status_lower:
            return ModuleName.NERVOUS_SYSTEM
        elif 'digital' in status_lower or 'sim' in status_lower:
            return ModuleName.DIGITAL_TWIN
        elif 'ai' in status_lower or 'brain' in status_lower or 'cognitive' in status_lower:
            return ModuleName.AI_BRAIN
        elif 'vla' in status_lower or 'task' in status_lower:
            return ModuleName.VLA_ROBOTICS
        elif 'integration' in status_lower or 'system' in status_lower:
            return ModuleName.INTEGRATION

        return None

    def estimate_health_from_status(self, status_msg: str) -> float:
        """Estimate health score from status message"""
        if 'error' in status_msg.lower():
            return 0.1
        elif 'warning' in status_msg.lower():
            return 0.5
        elif 'ok' in status_msg.lower() or 'ready' in status_msg.lower() or 'running' in status_msg.lower():
            return 0.9
        else:
            return 0.7

    def coordinator_command_callback(self, msg):
        """Callback for coordinator commands"""
        command = msg.data.lower().strip()

        if command == 'initialize_system':
            self.initialize_system()
        elif command == 'start_all_modules':
            self.start_all_modules()
        elif command == 'stop_all_modules':
            self.stop_all_modules()
        elif command.startswith('start_module:'):
            module_name_str = command.split(':')[1]
            self.start_module_by_name(module_name_str)
        elif command.startswith('stop_module:'):
            module_name_str = command.split(':')[1]
            self.stop_module_by_name(module_name_str)
        elif command == 'restart_system':
            self.restart_system()

    def coordinator_callback(self):
        """Main coordinator callback - manages module lifecycle"""
        if not self.coordinator_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_coordinator_time
        self.last_coordinator_time = current_time

        with self.module_lock:
            # Monitor module health
            self.monitor_module_health()

            # Check for module restart needs
            self.check_module_restart_needs()

        # Publish coordinator status
        self.publish_coordinator_status()

        # Publish system health
        avg_health = self.calculate_average_health()
        health_msg = Float32()
        health_msg.data = avg_health
        self.system_health_publisher.publish(health_msg)

    def monitor_module_health(self):
        """Monitor the health of all modules"""
        for module_name, module_info in self.modules.items():
            # Check if module is taking too long to respond
            time_since_update = time.time() - module_info.last_update

            if time_since_update > 10.0 and module_info.state == ModuleState.RUNNING:
                # Module hasn't updated in 10 seconds, potential issue
                self.get_logger().warn(f"Module {module_name.value} not responding for {time_since_update:.1f}s")
                module_info.health_score = max(0.1, module_info.health_score - 0.1)

    def check_module_restart_needs(self):
        """Check if any modules need to be restarted"""
        for module_name, module_info in self.modules.items():
            if module_info.health_score < 0.3 and module_info.state == ModuleState.RUNNING:
                self.get_logger().warn(f"Restarting module {module_name.value} due to low health ({module_info.health_score:.2f})")
                self.restart_module(module_name)

    def initialize_system(self):
        """Initialize the complete system"""
        self.get_logger().info('Initializing system...')

        # Start modules in dependency order
        for module_name in self.startup_sequence:
            if self.check_dependencies_ready(module_name):
                self.start_module(module_name)
                time.sleep(2)  # Wait 2 seconds between module starts

        self.system_initialized = True
        self.get_logger().info('System initialization complete')

    def start_all_modules(self):
        """Start all modules"""
        self.get_logger().info('Starting all modules...')

        for module_name in self.startup_sequence:
            if self.check_dependencies_ready(module_name):
                self.start_module(module_name)

        self.get_logger().info('All modules started')

    def stop_all_modules(self):
        """Stop all modules"""
        self.get_logger().info('Stopping all modules...')

        # Stop modules in reverse dependency order
        for module_name in self.shutdown_sequence:
            self.stop_module(module_name)

        self.get_logger().info('All modules stopped')

    def start_module_by_name(self, module_name_str: str):
        """Start a module by its string name"""
        try:
            module_name = ModuleName(module_name_str.upper().replace('-', '_'))
            self.start_module(module_name)
        except ValueError:
            self.get_logger().error(f"Unknown module name: {module_name_str}")

    def stop_module_by_name(self, module_name_str: str):
        """Stop a module by its string name"""
        try:
            module_name = ModuleName(module_name_str.upper().replace('-', '_'))
            self.stop_module(module_name)
        except ValueError:
            self.get_logger().error(f"Unknown module name: {module_name_str}")

    def start_module(self, module_name: ModuleName):
        """Start a specific module"""
        if module_name not in self.modules:
            self.get_logger().error(f"Unknown module: {module_name}")
            return False

        module_info = self.modules[module_name]

        if module_info.state in [ModuleState.STARTING, ModuleState.RUNNING]:
            self.get_logger().info(f"Module {module_name.value} is already starting or running")
            return True

        # Check dependencies
        if not self.check_dependencies_ready(module_name):
            self.get_logger().warn(f"Dependencies not ready for {module_name.value}, cannot start")
            return False

        try:
            # Update state
            module_info.state = ModuleState.STARTING

            # In a real system, this would launch the actual ROS 2 node
            # For simulation, we'll just update the state
            # cmd = ['ros2', 'run', module_name.value, module_info.launch_file]
            # process = subprocess.Popen(cmd)
            # module_info.pid = process.pid

            # For this simulation, we'll just set it to running after a delay
            time.sleep(0.5)  # Simulate startup time
            module_info.state = ModuleState.RUNNING
            module_info.health_score = 0.9  # Start with high health
            module_info.last_update = time.time()

            self.get_logger().info(f"Started module: {module_name.value}")

            # Publish control command
            cmd_msg = String()
            cmd_msg.data = f"STARTED: {module_name.value}"
            self.module_control_publisher.publish(cmd_msg)

            return True

        except Exception as e:
            self.get_logger().error(f"Failed to start module {module_name.value}: {e}")
            module_info.state = ModuleState.ERROR
            module_info.health_score = 0.1
            return False

    def stop_module(self, module_name: ModuleName):
        """Stop a specific module"""
        if module_name not in self.modules:
            self.get_logger().error(f"Unknown module: {module_name}")
            return False

        module_info = self.modules[module_name]

        if module_info.state in [ModuleState.STOPPING, ModuleState.STOPPED]:
            self.get_logger().info(f"Module {module_name.value} is already stopping or stopped")
            return True

        try:
            # Update state
            module_info.state = ModuleState.STOPPING

            # In a real system, this would terminate the actual ROS 2 node
            # For simulation, we'll just update the state
            # if module_info.pid:
            #     os.kill(module_info.pid, signal.SIGTERM)
            #     module_info.pid = None

            time.sleep(0.3)  # Simulate shutdown time
            module_info.state = ModuleState.STOPPED
            module_info.health_score = 0.0
            module_info.last_update = time.time()

            self.get_logger().info(f"Stopped module: {module_name.value}")

            # Publish control command
            cmd_msg = String()
            cmd_msg.data = f"STOPPED: {module_name.value}"
            self.module_control_publisher.publish(cmd_msg)

            return True

        except Exception as e:
            self.get_logger().error(f"Failed to stop module {module_name.value}: {e}")
            module_info.state = ModuleState.ERROR
            return False

    def restart_module(self, module_name: ModuleName):
        """Restart a specific module"""
        self.get_logger().info(f"Restarting module: {module_name.value}")

        success = self.stop_module(module_name)
        if success:
            time.sleep(1)  # Wait 1 second before restart
            success = self.start_module(module_name)

        return success

    def restart_system(self):
        """Restart the entire system"""
        self.get_logger().info('Restarting entire system...')

        self.stop_all_modules()
        time.sleep(2)
        self.start_all_modules()

    def check_dependencies_ready(self, module_name: ModuleName) -> bool:
        """Check if all dependencies for a module are ready"""
        module_info = self.modules[module_name]

        for dep_name in module_info.dependencies:
            dep_info = self.modules.get(dep_name)
            if not dep_info or dep_info.state != ModuleState.RUNNING:
                return False

        return True

    def publish_coordinator_status(self):
        """Publish coordinator status"""
        running_count = sum(1 for info in self.modules.values() if info.state == ModuleState.RUNNING)
        total_count = len(self.modules)

        status_parts = [
            f"Modules: {running_count}/{total_count} Running",
            f"Initialized: {self.system_initialized}",
            f"Coordinator: {'ENABLED' if self.coordinator_enabled else 'DISABLED'}"
        ]

        status_msg = String()
        status_msg.data = f"COORDINATOR: {', '.join(status_parts)}"
        self.coordinator_status_publisher.publish(status_msg)

    def calculate_average_health(self) -> float:
        """Calculate average health of all modules"""
        if not self.modules:
            return 0.0

        total_health = sum(info.health_score for info in self.modules.values())
        return total_health / len(self.modules)

    def enable_coordinator(self, enable: bool):
        """Enable or disable the module coordinator"""
        self.coordinator_enabled = enable
        self.get_logger().info(f"Module coordinator {'enabled' if enable else 'disabled'}")

    def get_module_status(self, module_name: ModuleName) -> Optional[Dict[str, any]]:
        """Get status of a specific module"""
        if module_name in self.modules:
            info = self.modules[module_name]
            return {
                'state': info.state.value,
                'health_score': info.health_score,
                'last_update': info.last_update,
                'dependencies': [dep.value for dep in info.dependencies]
            }
        return None

    def get_all_module_statuses(self) -> Dict[str, Dict[str, any]]:
        """Get statuses of all modules"""
        return {
            name.value: {
                'state': info.state.value,
                'health_score': info.health_score,
                'last_update': info.last_update,
                'dependencies': [dep.value for dep in info.dependencies]
            }
            for name, info in self.modules.items()
        }

    def get_coordinator_stats(self) -> Dict[str, any]:
        """Get coordinator statistics"""
        running_modules = sum(1 for info in self.modules.values() if info.state == ModuleState.RUNNING)
        avg_health = self.calculate_average_health()

        return {
            'total_modules': len(self.modules),
            'running_modules': running_modules,
            'average_health': avg_health,
            'system_initialized': self.system_initialized,
            'enabled': self.coordinator_enabled
        }


def main(args=None):
    rclpy.init(args=args)

    module_coordinator_node = ModuleCoordinatorNode()

    try:
        rclpy.spin(module_coordinator_node)
    except KeyboardInterrupt:
        # Stop all modules on shutdown
        module_coordinator_node.stop_all_modules()
        pass
    finally:
        module_coordinator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()