#!/usr/bin/env python3

"""
System Integration Node for Humanoid Robotics Platform

This node integrates all modules of the humanoid robotics system,
coordinating communication between the ROS 2 Nervous System,
Digital Twin Simulation, AI Brain (NVIDIA Isaac), and VLA Robotics modules.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from sensor_msgs.msg import JointState, Imu, LaserScan, Image
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
import time
import math
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum


class SystemModule(Enum):
    NERVOUS_SYSTEM = "nervous_system"
    DIGITAL_TWIN = "digital_twin"
    AI_BRAIN = "ai_brain"
    VLA_ROBOTICS = "vla_robotics"
    INTEGRATION = "integration"


class SystemState(Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    SAFETY_LOCKOUT = "safety_lockout"
    EMERGENCY_STOP = "emergency_stop"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ModuleStatus:
    """Represents the status of a system module"""
    name: str
    state: str
    health: float  # 0.0 to 1.0
    last_update: float
    enabled: bool


class SystemIntegrationNode(Node):
    def __init__(self):
        super().__init__('system_integration_node')

        # Publishers for system-wide communication
        self.system_status_publisher = self.create_publisher(String, 'system/status', 10)
        self.system_command_publisher = self.create_publisher(String, 'system/command', 10)
        self.emergency_stop_publisher = self.create_publisher(Bool, 'emergency_stop', 10)
        self.integration_status_publisher = self.create_publisher(String, 'integration/status', 10)

        # Subscribers for all module status updates
        self.nervous_system_status_subscriber = self.create_subscription(
            String, 'nervous_system/status', self.nervous_system_status_callback, 10)
        self.digital_twin_status_subscriber = self.create_subscription(
            String, 'digital_twin/status', self.digital_twin_status_callback, 10)
        self.ai_brain_status_subscriber = self.create_subscription(
            String, 'ai_brain/status', self.ai_brain_status_callback, 10)
        self.vla_status_subscriber = self.create_subscription(
            String, 'vla/status', self.vla_status_callback, 10)

        # Subscribers for critical system data
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Timer for system integration management
        self.integration_timer = self.create_timer(0.1, self.integration_callback)  # 10 Hz

        # System state management
        self.system_state = SystemState.INITIALIZING
        self.system_start_time = time.time()
        self.last_status_update = time.time()
        self.integration_enabled = True
        self.safety_enabled = True
        self.sim_time = time.time()

        # Module status tracking
        self.module_statuses: Dict[SystemModule, ModuleStatus] = {}
        self.initialize_module_statuses()

        # System data
        self.joint_states: Optional[JointState] = None
        self.imu_data: Optional[Imu] = None
        self.odom_data: Optional[Odometry] = None
        self.laser_data: Optional[LaserScan] = None

        # Safety monitoring
        self.safety_violations = []
        self.emergency_stop_active = False
        self.safety_thresholds = {
            'tilt_angle': 0.5,  # radians
            'collision_distance': 0.3,  # meters
            'joint_limits_violation': 0.1  # radians
        }

        # Performance metrics
        self.performance_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'communication_latency': 0.0,
            'module_sync_rate': 0.0
        }

        # Threading lock for data access
        self.data_lock = threading.Lock()

        self.get_logger().info('System Integration Node initialized')

    def initialize_module_statuses(self):
        """Initialize module status tracking"""
        for module in SystemModule:
            self.module_statuses[module] = ModuleStatus(
                name=module.value,
                state='unknown',
                health=0.5,
                last_update=0.0,
                enabled=True
            )

    def nervous_system_status_callback(self, msg):
        """Callback for nervous system status updates"""
        with self.data_lock:
            self.update_module_status(SystemModule.NERVOUS_SYSTEM, msg.data)

    def digital_twin_status_callback(self, msg):
        """Callback for digital twin status updates"""
        with self.data_lock:
            self.update_module_status(SystemModule.DIGITAL_TWIN, msg.data)

    def ai_brain_status_callback(self, msg):
        """Callback for AI brain status updates"""
        with self.data_lock:
            self.update_module_status(SystemModule.AI_BRAIN, msg.data)

    def vla_status_callback(self, msg):
        """Callback for VLA status updates"""
        with self.data_lock:
            self.update_module_status(SystemModule.VLA_ROBOTICS, msg.data)

    def update_module_status(self, module: SystemModule, status_str: str):
        """Update the status of a module based on status string"""
        # Parse status string (format: "STATUS: value1, value2, ...")
        parts = status_str.split(':')
        if len(parts) >= 2:
            state_info = parts[1].strip()
            health = self.estimate_health_from_status(state_info)

            self.module_statuses[module] = ModuleStatus(
                name=module.value,
                state=parts[0].strip(),
                health=health,
                last_update=time.time(),
                enabled=True
            )

    def estimate_health_from_status(self, status_info: str) -> float:
        """Estimate health value from status information"""
        # Simple health estimation based on status keywords
        if 'error' in status_info.lower() or 'failed' in status_info.lower():
            return 0.2
        elif 'warning' in status_info.lower():
            return 0.6
        elif 'ok' in status_info.lower() or 'ready' in status_info.lower():
            return 0.9
        else:
            return 0.7  # Default health

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        with self.data_lock:
            self.joint_states = msg

    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.data_lock:
            self.imu_data = msg

            # Check for safety violations
            if self.safety_enabled:
                self.check_balance_safety(msg)

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.data_lock:
            self.odom_data = msg

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        with self.data_lock:
            self.laser_data = msg

            # Check for safety violations
            if self.safety_enabled:
                self.check_collision_safety(msg)

    def check_balance_safety(self, imu_msg: Imu):
        """Check if robot is in a safe balance state"""
        # Calculate tilt angle from IMU orientation
        orientation = imu_msg.orientation
        # Simplified tilt calculation (in reality would use proper quaternion math)
        tilt_angle = math.sqrt(orientation.x**2 + orientation.y**2)

        if tilt_angle > self.safety_thresholds['tilt_angle']:
            violation = f"Balance violation: tilt_angle={tilt_angle:.3f} > threshold={self.safety_thresholds['tilt_angle']}"
            self.safety_violations.append((time.time(), violation))
            self.get_logger().warn(violation)

            # Trigger safety response
            self.trigger_safety_response('balance')

    def check_collision_safety(self, laser_msg: LaserScan):
        """Check if robot is approaching obstacles too closely"""
        if len(laser_msg.ranges) > 0:
            min_distance = min([r for r in laser_msg.ranges if not math.isnan(r)], default=float('inf'))

            if min_distance < self.safety_thresholds['collision_distance']:
                violation = f"Collision risk: distance={min_distance:.3f} < threshold={self.safety_thresholds['collision_distance']}"
                self.safety_violations.append((time.time(), violation))
                self.get_logger().warn(violation)

                # Trigger safety response
                self.trigger_safety_response('collision')

    def trigger_safety_response(self, violation_type: str):
        """Trigger appropriate safety response based on violation type"""
        if not self.safety_enabled:
            return

        if violation_type in ['balance', 'collision']:
            # Activate emergency stop
            self.emergency_stop_active = True
            self.system_state = SystemState.SAFETY_LOCKOUT

            # Publish emergency stop
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_publisher.publish(stop_msg)

            self.get_logger().error(f"SAFETY LOCKOUT ACTIVATED due to {violation_type} violation")

    def integration_callback(self):
        """Main integration callback - coordinates all system modules"""
        if not self.integration_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_status_update
        self.last_status_update = current_time

        with self.data_lock:
            # Update system state based on module statuses
            self.update_system_state()

            # Perform safety checks
            self.perform_safety_monitoring()

            # Coordinate module activities
            self.coordinate_modules()

            # Publish system status
            self.publish_system_status()

            # Clean up old safety violations
            self.cleanup_old_violations()

        # Update performance metrics
        self.update_performance_metrics()

    def update_system_state(self):
        """Update the overall system state based on module statuses"""
        # Check if all critical modules are operational
        critical_modules = [
            SystemModule.NERVOUS_SYSTEM,
            SystemModule.AI_BRAIN
        ]

        all_operational = True
        for module in critical_modules:
            status = self.module_statuses.get(module)
            if not status or status.health < 0.3 or 'error' in status.state.lower():
                all_operational = False
                break

        if self.emergency_stop_active:
            self.system_state = SystemState.EMERGENCY_STOP
        elif not all_operational:
            self.system_state = SystemState.SAFETY_LOCKOUT
        elif self.system_state == SystemState.INITIALIZING:
            # Check if initialization is complete
            all_initialized = all(
                'ready' in self.module_statuses[mod].state.lower() or
                'ok' in self.module_statuses[mod].state.lower()
                for mod in SystemModule
            )
            if all_initialized:
                self.system_state = SystemModule.ACTIVE
        else:
            self.system_state = SystemState.ACTIVE

    def perform_safety_monitoring(self):
        """Perform continuous safety monitoring"""
        # Monitor joint limits if joint states are available
        if self.joint_states:
            self.check_joint_limits()

        # Monitor overall system health
        avg_health = self.calculate_average_module_health()
        if avg_health < 0.3:  # System health below threshold
            self.get_logger().warn(f"Low system health: {avg_health:.2f}")
            if not self.emergency_stop_active:
                self.system_state = SystemState.SAFETY_LOCKOUT

    def check_joint_limits(self):
        """Check if any joints are violating limits"""
        # This would check actual joint limits in a real system
        # For now, we'll just log the joint states
        pass

    def coordinate_modules(self):
        """Coordinate activities between modules"""
        # Example coordination: If AI Brain requests navigation,
        # ensure nervous system is ready to execute commands
        if (self.module_statuses[SystemModule.AI_BRAIN].state == 'navigation_request' and
            self.module_statuses[SystemModule.NERVOUS_SYSTEM].state == 'ready'):
            # Send coordination command
            cmd_msg = String()
            cmd_msg.data = "COORDINATE: ai_brain_navigate -> nervous_system_execute"
            self.system_command_publisher.publish(cmd_msg)

        # Example: If VLA system has a task, coordinate with AI Brain
        if (self.module_statuses[SystemModule.VLA_ROBOTICS].state == 'task_ready' and
            self.module_statuses[SystemModule.AI_BRAIN].state == 'ready'):
            cmd_msg = String()
            cmd_msg.data = "COORDINATE: vla_task -> ai_brain_plan"
            self.system_command_publisher.publish(cmd_msg)

    def publish_system_status(self):
        """Publish overall system status"""
        status_parts = [
            f"System: {self.system_state.value}",
            f"Modules: {len([s for s in self.module_statuses.values() if s.health > 0.5])}/{len(self.module_statuses)} OK",
            f"Uptime: {time.time() - self.system_start_time:.1f}s",
            f"Violations: {len(self.safety_violations)}",
            f"Emergency: {'ACTIVE' if self.emergency_stop_active else 'INACTIVE'}"
        ]

        status_msg = String()
        status_msg.data = f"SYSTEM: {', '.join(status_parts)}"
        self.system_status_publisher.publish(status_msg)

        # Also publish integration-specific status
        integration_msg = String()
        integration_msg.data = f"INTEGRATION: State={self.system_state.value}, Modules={len(self.module_statuses)}, Time={time.time():.2f}"
        self.integration_status_publisher.publish(integration_msg)

    def cleanup_old_violations(self):
        """Clean up safety violations older than 10 seconds"""
        current_time = time.time()
        self.safety_violations = [
            (t, v) for t, v in self.safety_violations
            if current_time - t < 10.0
        ]

    def calculate_average_module_health(self) -> float:
        """Calculate the average health of all modules"""
        if not self.module_statuses:
            return 0.0

        total_health = sum(status.health for status in self.module_statuses.values())
        return total_health / len(self.module_statuses)

    def enable_integration(self, enable: bool):
        """Enable or disable system integration"""
        self.integration_enabled = enable
        self.get_logger().info(f"System integration {'enabled' if enable else 'disabled'}")

    def enable_safety(self, enable: bool):
        """Enable or disable safety monitoring"""
        self.safety_enabled = enable
        self.get_logger().info(f"Safety monitoring {'enabled' if enable else 'disabled'}")

    def trigger_emergency_stop(self):
        """Manually trigger emergency stop"""
        self.emergency_stop_active = True
        self.system_state = SystemState.EMERGENCY_STOP

        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_publisher.publish(stop_msg)

        self.get_logger().error("Manual emergency stop triggered")

    def clear_emergency_stop(self):
        """Clear emergency stop condition"""
        self.emergency_stop_active = False
        self.system_state = SystemState.ACTIVE

        stop_msg = Bool()
        stop_msg.data = False
        self.emergency_stop_publisher.publish(stop_msg)

        self.get_logger().info("Emergency stop cleared")

    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status"""
        return {
            'system_state': self.system_state.value,
            'uptime': time.time() - self.system_start_time,
            'integration_enabled': self.integration_enabled,
            'safety_enabled': self.safety_enabled,
            'emergency_stop_active': self.emergency_stop_active,
            'module_statuses': {k.value: {
                'state': v.state,
                'health': v.health,
                'last_update': v.last_update
            } for k, v in self.module_statuses.items()},
            'safety_violations_count': len(self.safety_violations),
            'average_module_health': self.calculate_average_module_health()
        }

    def update_performance_metrics(self):
        """Update system performance metrics"""
        # In a real system, this would gather actual performance data
        # For simulation, we'll use placeholder values
        self.performance_metrics['cpu_usage'] = 0.45  # 45% CPU usage
        self.performance_metrics['memory_usage'] = 0.60  # 60% memory usage
        self.performance_metrics['communication_latency'] = 0.02  # 20ms latency
        self.performance_metrics['module_sync_rate'] = 0.95  # 95% sync rate


def main(args=None):
    rclpy.init(args=args)

    system_integration_node = SystemIntegrationNode()

    try:
        rclpy.spin(system_integration_node)
    except KeyboardInterrupt:
        # Trigger emergency stop on shutdown for safety
        system_integration_node.trigger_emergency_stop()
        pass
    finally:
        system_integration_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()