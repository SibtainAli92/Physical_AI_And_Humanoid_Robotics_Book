#!/usr/bin/env python3

"""
Safety Manager Node for Humanoid Robotics Platform

This node manages safety protocols across the entire integrated system,
monitoring for hazards and implementing safety responses.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import time
import math
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import json


class SafetyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALERT = "alert"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class SafetyZone(Enum):
    WORKSPACE = "workspace"
    HUMAN_PROXIMITY = "human_proximity"
    COLLISION_RISK = "collision_risk"
    BALANCE_CRITICAL = "balance_critical"
    JOINT_LIMIT = "joint_limit"


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    id: str
    level: SafetyLevel
    zone: SafetyZone
    description: str
    timestamp: float
    severity: float  # 0.0 to 1.0
    active: bool


class SafetyManagerNode(Node):
    def __init__(self):
        super().__init__('safety_manager_node')

        # Publishers for safety management
        self.safety_status_publisher = self.create_publisher(String, 'safety/status', 10)
        self.emergency_stop_publisher = self.create_publisher(Bool, 'emergency_stop', 10)
        self.safety_command_publisher = self.create_publisher(String, 'safety/command', 10)
        self.safety_violation_publisher = self.create_publisher(String, 'safety/violations', 10)

        # Subscribers for safety-critical data
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.twist_subscriber = self.create_subscription(
            Twist, 'motion_commands', self.twist_callback, 10)

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.05, self.safety_monitoring_callback)  # 20 Hz

        # Safety management state
        self.safety_enabled = True
        self.emergency_stop_active = False
        self.safety_level = SafetyLevel.NORMAL
        self.active_violations = []
        self.violation_history = []
        self.sim_time = time.time()
        self.last_safety_time = time.time()

        # Robot state tracking
        self.joint_states: Optional[JointState] = None
        self.imu_data: Optional[Imu] = None
        self.odom_data: Optional[Odometry] = None
        self.laser_data: Optional[LaserScan] = None
        self.current_twist: Optional[Twist] = None
        self.current_pose: Optional[Pose] = None

        # Safety thresholds
        self.safety_thresholds = {
            'tilt_angle': 0.5,  # radians
            'collision_distance': 0.3,  # meters
            'max_velocity': 0.5,  # m/s
            'max_angular_velocity': 0.8,  # rad/s
            'joint_limit_margin': 0.1,  # radians
            'proximity_distance': 1.0,  # meters for human detection
            'acceleration_limit': 2.0  # m/s^2
        }

        # Joint limits (example values - would be robot-specific)
        self.joint_limits = {
            'hip_joint': (-1.57, 1.57),
            'knee_joint': (0, 2.35),
            'ankle_joint': (-0.78, 0.78),
            'shoulder_joint': (-1.57, 1.57),
            'elbow_joint': (0, 2.35),
        }

        # Safety monitoring
        self.last_joint_states = None
        self.last_imu_data = None
        self.last_timestamp = time.time()

        # Threading lock for safety data
        self.safety_lock = threading.Lock()

        self.get_logger().info('Safety Manager Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        with self.safety_lock:
            self.last_joint_states = self.joint_states
            self.joint_states = msg
            self.check_joint_safety()

    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.safety_lock:
            self.last_imu_data = self.imu_data
            self.imu_data = msg
            self.check_balance_safety()

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.safety_lock:
            self.odom_data = msg
            self.current_pose = msg.pose.pose

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        with self.safety_lock:
            self.laser_data = msg
            self.check_collision_safety()

    def twist_callback(self, msg):
        """Callback for motion commands"""
        with self.safety_lock:
            self.current_twist = msg
            self.check_motion_safety()

    def safety_monitoring_callback(self):
        """Main safety monitoring callback"""
        if not self.safety_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_safety_time
        self.last_safety_time = current_time

        with self.safety_lock:
            # Update safety level based on active violations
            self.update_safety_level()

            # Check for safety recovery
            self.check_safety_recovery()

            # Clean up old violations
            self.cleanup_old_violations()

        # Publish safety status
        self.publish_safety_status()

        # Publish active violations
        self.publish_active_violations()

    def check_joint_safety(self):
        """Check for joint-related safety violations"""
        if not self.joint_states:
            return

        for i, joint_name in enumerate(self.joint_states.name):
            if i < len(self.joint_states.position):
                position = self.joint_states.position[i]

                if joint_name in self.joint_limits:
                    min_limit, max_limit = self.joint_limits[joint_name]

                    # Check if approaching limits
                    if position < min_limit + self.safety_thresholds['joint_limit_margin']:
                        violation = SafetyViolation(
                            id=f"joint_limit_{joint_name}_low_{int(time.time())}",
                            level=SafetyLevel.WARNING,
                            zone=SafetyZone.JOINT_LIMIT,
                            description=f"Joint {joint_name} approaching lower limit: {position:.3f} < {min_limit + self.safety_thresholds['joint_limit_margin']:.3f}",
                            timestamp=time.time(),
                            severity=0.3,
                            active=True
                        )
                        self.add_violation(violation)

                    elif position > max_limit - self.safety_thresholds['joint_limit_margin']:
                        violation = SafetyViolation(
                            id=f"joint_limit_{joint_name}_high_{int(time.time())}",
                            level=SafetyLevel.WARNING,
                            zone=SafetyZone.JOINT_LIMIT,
                            description=f"Joint {joint_name} approaching upper limit: {position:.3f} > {max_limit - self.safety_thresholds['joint_limit_margin']:.3f}",
                            timestamp=time.time(),
                            severity=0.3,
                            active=True
                        )
                        self.add_violation(violation)

                    # Check if exceeding limits (emergency)
                    if position < min_limit or position > max_limit:
                        violation = SafetyViolation(
                            id=f"joint_limit_exceeded_{joint_name}_{int(time.time())}",
                            level=SafetyLevel.EMERGENCY,
                            zone=SafetyZone.JOINT_LIMIT,
                            description=f"Joint {joint_name} limit exceeded: {position:.3f}",
                            timestamp=time.time(),
                            severity=0.9,
                            active=True
                        )
                        self.add_violation(violation)

    def check_balance_safety(self):
        """Check for balance-related safety violations"""
        if not self.imu_data:
            return

        # Calculate tilt angle from IMU orientation
        orientation = self.imu_data.orientation
        # Simplified tilt calculation
        tilt_angle = math.sqrt(orientation.x**2 + orientation.y**2)

        if tilt_angle > self.safety_thresholds['tilt_angle']:
            severity = min(1.0, (tilt_angle - self.safety_thresholds['tilt_angle']) / 0.5)
            level = SafetyLevel.EMERGENCY if severity > 0.7 else SafetyLevel.ALERT

            violation = SafetyViolation(
                id=f"balance_violation_{int(time.time())}",
                level=level,
                zone=SafetyZone.BALANCE_CRITICAL,
                description=f"Balance angle too high: {tilt_angle:.3f} > {self.safety_thresholds['tilt_angle']:.3f}",
                timestamp=time.time(),
                severity=severity,
                active=True
            )
            self.add_violation(violation)

    def check_collision_safety(self):
        """Check for collision-related safety violations"""
        if not self.laser_data or len(self.laser_data.ranges) == 0:
            return

        min_distance = min([r for r in self.laser_data.ranges if not math.isnan(r)], default=float('inf'))

        if min_distance < self.safety_thresholds['collision_distance']:
            severity = 1.0 - (min_distance / self.safety_thresholds['collision_distance'])
            level = SafetyLevel.EMERGENCY if severity > 0.8 else SafetyLevel.ALERT

            violation = SafetyViolation(
                id=f"collision_risk_{int(time.time())}",
                level=level,
                zone=SafetyZone.COLLISION_RISK,
                description=f"Collision risk: distance {min_distance:.3f} < threshold {self.safety_thresholds['collision_distance']:.3f}",
                timestamp=time.time(),
                severity=severity,
                active=True
            )
            self.add_violation(violation)

        elif min_distance < self.safety_thresholds['proximity_distance']:
            # Warning for proximity but not immediate collision risk
            violation = SafetyViolation(
                id=f"proximity_warning_{int(time.time())}",
                level=SafetyLevel.WARNING,
                zone=SafetyZone.HUMAN_PROXIMITY,
                description=f"Object proximity: distance {min_distance:.3f} < {self.safety_thresholds['proximity_distance']:.3f}",
                timestamp=time.time(),
                severity=0.2,
                active=True
            )
            self.add_violation(violation)

    def check_motion_safety(self):
        """Check for motion-related safety violations"""
        if not self.current_twist:
            return

        # Check linear velocity limits
        linear_speed = math.sqrt(
            self.current_twist.linear.x**2 +
            self.current_twist.linear.y**2 +
            self.current_twist.linear.z**2
        )

        if linear_speed > self.safety_thresholds['max_velocity']:
            severity = min(1.0, (linear_speed - self.safety_thresholds['max_velocity']) / 0.5)
            level = SafetyLevel.EMERGENCY if severity > 0.7 else SafetyLevel.ALERT

            violation = SafetyViolation(
                id=f"speed_limit_{int(time.time())}",
                level=level,
                zone=SafetyZone.WORKSPACE,
                description=f"Speed limit exceeded: {linear_speed:.3f} > {self.safety_thresholds['max_velocity']:.3f}",
                timestamp=time.time(),
                severity=severity,
                active=True
            )
            self.add_violation(violation)

        # Check angular velocity limits
        angular_speed = math.sqrt(
            self.current_twist.angular.x**2 +
            self.current_twist.angular.y**2 +
            self.current_twist.angular.z**2
        )

        if angular_speed > self.safety_thresholds['max_angular_velocity']:
            severity = min(1.0, (angular_speed - self.safety_thresholds['max_angular_velocity']) / 0.5)
            level = SafetyLevel.EMERGENCY if severity > 0.7 else SafetyLevel.ALERT

            violation = SafetyViolation(
                id=f"angular_speed_limit_{int(time.time())}",
                level=level,
                zone=SafetyZone.WORKSPACE,
                description=f"Angular speed limit exceeded: {angular_speed:.3f} > {self.safety_thresholds['max_angular_velocity']:.3f}",
                timestamp=time.time(),
                severity=severity,
                active=True
            )
            self.add_violation(violation)

    def add_violation(self, violation: SafetyViolation):
        """Add a safety violation to the active list"""
        # Check if this is a duplicate of an active violation
        for active_violation in self.active_violations:
            if (active_violation.zone == violation.zone and
                active_violation.description == violation.description and
                time.time() - active_violation.timestamp < 1.0):  # Same violation in last second
                return  # Skip duplicate

        self.active_violations.append(violation)
        self.violation_history.append(violation)

        # Log the violation
        self.get_logger().warn(f"Safety Violation: {violation.level.value} - {violation.description}")

        # Trigger appropriate response based on severity
        if violation.level in [SafetyLevel.EMERGENCY, SafetyLevel.CRITICAL]:
            self.trigger_emergency_response()
        elif violation.level == SafetyLevel.ALERT:
            self.trigger_alert_response()

    def update_safety_level(self):
        """Update the overall safety level based on active violations"""
        if not self.active_violations:
            self.safety_level = SafetyLevel.NORMAL
            return

        # Determine highest level among active violations
        highest_level = SafetyLevel.NORMAL
        for violation in self.active_violations:
            if (violation.level == SafetyLevel.CRITICAL or
                (violation.level == SafetyLevel.EMERGENCY and highest_level != SafetyLevel.CRITICAL) or
                (violation.level == SafetyLevel.ALERT and highest_level not in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]) or
                (violation.level == SafetyLevel.WARNING and highest_level == SafetyLevel.NORMAL)):
                highest_level = violation.level

        self.safety_level = highest_level

    def check_safety_recovery(self):
        """Check if safety conditions have improved"""
        if self.safety_level != SafetyLevel.NORMAL and not self.active_violations:
            # All violations cleared, but need to check if we can clear emergency stop
            if self.emergency_stop_active:
                # Check if all safety conditions are now OK
                recovery_possible = True

                # Check balance
                if self.imu_data:
                    tilt_angle = math.sqrt(self.imu_data.orientation.x**2 + self.imu_data.orientation.y**2)
                    if tilt_angle > self.safety_thresholds['tilt_angle'] * 0.7:  # 70% of threshold
                        recovery_possible = False

                # Check for obstacles
                if self.laser_data:
                    min_distance = min([r for r in self.laser_data.ranges if not math.isnan(r)], default=float('inf'))
                    if min_distance < self.safety_thresholds['collision_distance'] * 1.5:  # 150% of threshold
                        recovery_possible = False

                if recovery_possible:
                    self.clear_emergency_stop()
                    self.get_logger().info("Safety conditions recovered, clearing emergency stop")

    def cleanup_old_violations(self):
        """Clean up violations older than 5 seconds"""
        current_time = time.time()
        self.active_violations = [
            v for v in self.active_violations
            if current_time - v.timestamp < 5.0
        ]

        # Keep only recent history (last 100 violations)
        if len(self.violation_history) > 100:
            self.violation_history = self.violation_history[-100:]

    def trigger_emergency_response(self):
        """Trigger emergency safety response"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.safety_level = SafetyLevel.EMERGENCY

            # Publish emergency stop
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_publisher.publish(stop_msg)

            # Publish emergency command
            cmd_msg = String()
            cmd_msg.data = "EMERGENCY_STOP: All motion stopped due to safety violation"
            self.safety_command_publisher.publish(cmd_msg)

            self.get_logger().error("EMERGENCY STOP ACTIVATED")

    def trigger_alert_response(self):
        """Trigger alert-level safety response"""
        # For alert level, we might slow down or change behavior
        # but not stop completely
        cmd_msg = String()
        cmd_msg.data = "ALERT: Safety monitoring active, reducing speed"
        self.safety_command_publisher.publish(cmd_msg)

        self.get_logger().warn("Safety alert - monitoring and adjusting behavior")

    def clear_emergency_stop(self):
        """Clear emergency stop condition"""
        self.emergency_stop_active = False
        self.safety_level = SafetyLevel.NORMAL

        # Publish clear emergency stop
        stop_msg = Bool()
        stop_msg.data = False
        self.emergency_stop_publisher.publish(stop_msg)

        # Publish clear command
        cmd_msg = String()
        cmd_msg.data = "EMERGENCY_STOP_CLEARED: Safety conditions improved"
        self.safety_command_publisher.publish(cmd_msg)

        self.get_logger().info("Emergency stop cleared - safety conditions improved")

    def publish_safety_status(self):
        """Publish safety status"""
        active_count = len(self.active_violations)
        total_count = len(self.violation_history)

        status_parts = [
            f"Level: {self.safety_level.value}",
            f"Active Violations: {active_count}",
            f"Total Violations: {total_count}",
            f"Emergency: {'ACTIVE' if self.emergency_stop_active else 'INACTIVE'}",
            f"Enabled: {'YES' if self.safety_enabled else 'NO'}"
        ]

        status_msg = String()
        status_msg.data = f"SAFETY: {', '.join(status_parts)}"
        self.safety_status_publisher.publish(status_msg)

    def publish_active_violations(self):
        """Publish active violations"""
        if not self.active_violations:
            return

        # Create a summary of active violations
        violation_summary = {
            'timestamp': time.time(),
            'active_count': len(self.active_violations),
            'violations': [
                {
                    'id': v.id,
                    'level': v.level.value,
                    'zone': v.zone.value,
                    'description': v.description,
                    'severity': v.severity
                }
                for v in self.active_violations
            ]
        }

        violation_msg = String()
        violation_msg.data = json.dumps(violation_summary)
        self.safety_violation_publisher.publish(violation_msg)

    def enable_safety_system(self, enable: bool):
        """Enable or disable the safety system"""
        self.safety_enabled = enable
        self.get_logger().info(f"Safety system {'enabled' if enable else 'disabled'}")

        if not enable and self.emergency_stop_active:
            # If disabling safety, clear emergency stop
            self.clear_emergency_stop()

    def get_safety_status(self) -> Dict[str, any]:
        """Get comprehensive safety status"""
        return {
            'safety_level': self.safety_level.value,
            'emergency_stop_active': self.emergency_stop_active,
            'active_violations': len(self.active_violations),
            'total_violations': len(self.violation_history),
            'enabled': self.safety_enabled,
            'last_violation_time': max([v.timestamp for v in self.violation_history], default=0) if self.violation_history else 0
        }

    def get_active_violations(self) -> List[Dict[str, any]]:
        """Get active safety violations"""
        return [
            {
                'id': v.id,
                'level': v.level.value,
                'zone': v.zone.value,
                'description': v.description,
                'severity': v.severity,
                'timestamp': v.timestamp
            }
            for v in self.active_violations
        ]

    def reset_safety_system(self):
        """Reset the safety system"""
        with self.safety_lock:
            self.active_violations.clear()
            self.violation_history.clear()
            self.safety_level = SafetyLevel.NORMAL

            if self.emergency_stop_active:
                self.clear_emergency_stop()

            self.get_logger().info("Safety system reset")


def main(args=None):
    rclpy.init(args=args)

    safety_manager_node = SafetyManagerNode()

    try:
        rclpy.spin(safety_manager_node)
    except KeyboardInterrupt:
        # Ensure safety on shutdown
        safety_manager_node.trigger_emergency_response()
        pass
    finally:
        safety_manager_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()