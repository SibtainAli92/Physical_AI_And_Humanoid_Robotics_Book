#!/usr/bin/env python3

"""
Behavior Coordinator Node for Humanoid Robotics Platform

This node coordinates high-level behaviors across the integrated system,
managing behavior transitions and ensuring smooth operation between modules.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
import time
import math
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import json


class BehaviorState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    INTERACTING = "interacting"
    SPEAKING = "speaking"
    LISTENING = "listening"
    WAITING = "waiting"
    EMERGENCY_STOP = "emergency_stop"
    SAFETY_RESPONSE = "safety_response"


class BehaviorPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BehaviorRequest:
    """Represents a behavior request"""
    id: str
    behavior: BehaviorState
    priority: BehaviorPriority
    parameters: Dict[str, any]
    created_time: float
    timeout: float
    source: str


@dataclass
class ActiveBehavior:
    """Represents an active behavior"""
    id: str
    behavior: BehaviorState
    priority: BehaviorPriority
    parameters: Dict[str, any]
    start_time: float
    progress: float  # 0.0 to 1.0
    status: str  # 'active', 'completing', 'failed'


class BehaviorCoordinatorNode(Node):
    def __init__(self):
        super().__init__('behavior_coordinator_node')

        # Publishers for behavior coordination
        self.behavior_status_publisher = self.create_publisher(String, 'behavior/status', 10)
        self.behavior_command_publisher = self.create_publisher(String, 'behavior/command', 10)
        self.motion_command_publisher = self.create_publisher(Twist, 'motion_commands', 10)
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)

        # Subscribers for behavior-related data
        self.behavior_request_subscriber = self.create_subscription(
            String, 'behavior/request', self.behavior_request_callback, 10)
        self.ai_decision_subscriber = self.create_subscription(
            String, 'ai/decision', self.ai_decision_callback, 10)
        self.vla_task_subscriber = self.create_subscription(
            String, 'vla/task_status', self.vla_task_callback, 10)
        self.safety_status_subscriber = self.create_subscription(
            String, 'safety/status', self.safety_status_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Timer for behavior coordination
        self.behavior_timer = self.create_timer(0.1, self.behavior_coordination_callback)  # 10 Hz

        # Behavior coordination state
        self.behavior_requests = []
        self.active_behaviors = []
        self.current_behavior: Optional[ActiveBehavior] = None
        self.behavior_queue = []
        self.behavior_history = []
        self.behavior_coordination_enabled = True
        self.sim_time = time.time()
        self.last_behavior_time = time.time()

        # Robot state
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.current_imu = None
        self.balance_stable = True

        # Behavior parameters
        self.navigation_speed = 0.3
        self.rotation_speed = 0.5
        self.manipulation_speed = 0.1
        self.behavior_timeout = 30.0

        # Behavior transition rules
        self.transition_rules = {
            BehaviorState.IDLE: [BehaviorState.NAVIGATING, BehaviorState.MANIPULATING, BehaviorState.INTERACTING, BehaviorState.SPEAKING],
            BehaviorState.NAVIGATING: [BehaviorState.IDLE, BehaviorState.WAITING, BehaviorState.EMERGENCY_STOP],
            BehaviorState.MANIPULATING: [BehaviorState.IDLE, BehaviorState.NAVIGATING, BehaviorState.WAITING, BehaviorState.EMERGENCY_STOP],
            BehaviorState.INTERACTING: [BehaviorState.IDLE, BehaviorState.SPEAKING, BehaviorState.WAITING, BehaviorState.EMERGENCY_STOP],
            BehaviorState.SPEAKING: [BehaviorState.IDLE, BehaviorState.INTERACTING, BehaviorState.WAITING],
            BehaviorState.WAITING: [BehaviorState.IDLE, BehaviorState.NAVIGATING, BehaviorState.MANIPULATING],
            BehaviorState.EMERGENCY_STOP: [BehaviorState.IDLE],  # Can only go to idle after emergency
            BehaviorState.SAFETY_RESPONSE: [BehaviorState.IDLE, BehaviorState.EMERGENCY_STOP]
        }

        # Safety state
        self.safety_emergency_active = False

        # Threading lock for behavior management
        self.behavior_lock = threading.Lock()

        self.get_logger().info('Behavior Coordinator Node initialized')

    def behavior_request_callback(self, msg):
        """Callback for behavior requests"""
        try:
            # Parse behavior request from JSON
            request_data = json.loads(msg.data)

            behavior_request = BehaviorRequest(
                id=request_data['id'],
                behavior=BehaviorState(request_data['behavior']),
                priority=BehaviorPriority(request_data.get('priority', BehaviorPriority.MEDIUM.value)),
                parameters=request_data.get('parameters', {}),
                created_time=time.time(),
                timeout=request_data.get('timeout', self.behavior_timeout),
                source=request_data.get('source', 'unknown')
            )

            with self.behavior_lock:
                # Add to requests queue
                self.behavior_requests.append(behavior_request)

                # Sort by priority (highest first)
                self.behavior_requests.sort(key=lambda x: x.priority.value, reverse=True)

                self.get_logger().info(f"Received behavior request: {behavior_request.behavior.value} (Priority: {behavior_request.priority.name})")

        except json.JSONDecodeError:
            # If not JSON, try to parse as simple command
            with self.behavior_lock:
                # Determine behavior from simple string
                behavior = self.determine_behavior_from_command(msg.data)
                request = BehaviorRequest(
                    id=f"simple_{int(time.time())}",
                    behavior=behavior,
                    priority=BehaviorPriority.MEDIUM,
                    parameters={'command': msg.data},
                    created_time=time.time(),
                    timeout=self.behavior_timeout,
                    source='simple_command'
                )
                self.behavior_requests.append(request)

    def ai_decision_callback(self, msg):
        """Callback for AI decisions that may trigger behaviors"""
        # Process AI decision and potentially create behavior request
        pass

    def vla_task_callback(self, msg):
        """Callback for VLA task status updates"""
        # Process VLA task status and update behavior accordingly
        pass

    def safety_status_callback(self, msg):
        """Callback for safety status updates"""
        if 'EMERGENCY' in msg.data or 'CRITICAL' in msg.data:
            self.safety_emergency_active = True
            with self.behavior_lock:
                if self.current_behavior and self.current_behavior.behavior != BehaviorState.EMERGENCY_STOP:
                    self.trigger_behavior_transition(BehaviorState.EMERGENCY_STOP)
        else:
            self.safety_emergency_active = False

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.behavior_lock:
            self.current_pose = msg.pose.pose
            self.current_twist = msg.twist.twist

    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.behavior_lock:
            self.current_imu = msg
            # Check balance stability
            tilt_threshold = 0.3
            tilt_magnitude = math.sqrt(msg.orientation.x**2 + msg.orientation.y**2)
            self.balance_stable = tilt_magnitude < tilt_threshold

    def behavior_coordination_callback(self):
        """Main behavior coordination callback"""
        if not self.behavior_coordination_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_behavior_time
        self.last_behavior_time = current_time

        with self.behavior_lock:
            # Process behavior requests
            self.process_behavior_requests()

            # Execute current behavior
            if self.current_behavior:
                self.execute_current_behavior()

            # Check for behavior completion or timeouts
            self.check_behavior_timeouts()

            # Update behavior history
            self.update_behavior_history()

        # Publish behavior status
        self.publish_behavior_status()

    def process_behavior_requests(self):
        """Process pending behavior requests"""
        for request in self.behavior_requests[:]:  # Use slice to avoid modification during iteration
            if time.time() > request.created_time + request.timeout:
                # Request timed out
                self.behavior_requests.remove(request)
                continue

            # Check if this request should preempt current behavior
            if self.should_preempt_current_behavior(request):
                self.preempt_current_behavior(request)
            elif not self.current_behavior or self.current_behavior.behavior == BehaviorState.IDLE:
                # Start new behavior if no current behavior or idle
                self.start_behavior_from_request(request)
                if request in self.behavior_requests:
                    self.behavior_requests.remove(request)

    def should_preempt_current_behavior(self, request: BehaviorRequest) -> bool:
        """Determine if a request should preempt the current behavior"""
        if not self.current_behavior:
            return False

        # Critical and emergency behaviors can preempt anything
        if request.priority == BehaviorPriority.CRITICAL or request.behavior == BehaviorState.EMERGENCY_STOP:
            return True

        # Higher priority requests can preempt lower priority behaviors
        if request.priority.value > self.current_behavior.priority.value:
            return True

        # Safety-related behaviors can preempt most others
        if self.safety_emergency_active and request.behavior in [BehaviorState.EMERGENCY_STOP, BehaviorState.SAFETY_RESPONSE]:
            return True

        return False

    def preempt_current_behavior(self, request: BehaviorRequest):
        """Preempt the current behavior with a new one"""
        if self.current_behavior:
            # Log preemption
            self.get_logger().info(f"Preempting {self.current_behavior.behavior.value} with {request.behavior.value}")

            # Add current behavior to history as interrupted
            self.current_behavior.status = 'interrupted'
            self.behavior_history.append(self.current_behavior)

        # Start the new behavior
        self.start_behavior_from_request(request)

        # Remove from requests
        if request in self.behavior_requests:
            self.behavior_requests.remove(request)

    def start_behavior_from_request(self, request: BehaviorRequest):
        """Start a behavior based on a request"""
        # Check if transition is allowed
        if self.current_behavior:
            if not self.is_valid_transition(self.current_behavior.behavior, request.behavior):
                self.get_logger().warn(f"Invalid behavior transition: {self.current_behavior.behavior.value} -> {request.behavior.value}")
                return

        # Create new active behavior
        active_behavior = ActiveBehavior(
            id=f"active_{request.id}",
            behavior=request.behavior,
            priority=request.priority,
            parameters=request.parameters,
            start_time=time.time(),
            progress=0.0,
            status='active'
        )

        self.current_behavior = active_behavior

        # Log behavior start
        self.get_logger().info(f"Starting behavior: {request.behavior.value}")

        # Publish command to start behavior
        cmd_msg = String()
        cmd_msg.data = f"BEHAVIOR_START: {request.behavior.value}"
        self.behavior_command_publisher.publish(cmd_msg)

    def is_valid_transition(self, from_behavior: BehaviorState, to_behavior: BehaviorState) -> bool:
        """Check if a behavior transition is valid"""
        if from_behavior in self.transition_rules:
            return to_behavior in self.transition_rules[from_behavior]
        return False

    def execute_current_behavior(self):
        """Execute the current behavior"""
        if not self.current_behavior:
            return

        behavior = self.current_behavior.behavior

        if behavior == BehaviorState.NAVIGATING:
            self.execute_navigation_behavior()
        elif behavior == BehaviorState.MANIPULATING:
            self.execute_manipulation_behavior()
        elif behavior == BehaviorState.INTERACTING:
            self.execute_interaction_behavior()
        elif behavior == BehaviorState.SPEAKING:
            self.execute_speaking_behavior()
        elif behavior == BehaviorState.WAITING:
            self.execute_waiting_behavior()
        elif behavior == BehaviorState.EMERGENCY_STOP:
            self.execute_emergency_stop_behavior()
        elif behavior == BehaviorState.SAFETY_RESPONSE:
            self.execute_safety_response_behavior()
        else:
            # Default to idle behavior
            self.execute_idle_behavior()

    def execute_navigation_behavior(self):
        """Execute navigation behavior"""
        params = self.current_behavior.parameters

        # Extract target location
        target_x = params.get('target_x', self.current_pose.position.x)
        target_y = params.get('target_y', self.current_pose.position.y)

        # Calculate distance to target
        dx = target_x - self.current_pose.position.x
        dy = target_y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if arrived
        arrival_threshold = 0.2  # meters
        if distance < arrival_threshold:
            self.complete_current_behavior()
            return

        # Generate motion command
        cmd = Twist()
        if distance > 0:
            cmd.linear.x = dx / distance * self.navigation_speed
            cmd.linear.y = dy / distance * self.navigation_speed

        # Add rotation if needed
        target_yaw = math.atan2(dy, dx)
        current_yaw = self.get_current_yaw()
        yaw_diff = target_yaw - current_yaw

        # Normalize yaw difference
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi

        if abs(yaw_diff) > 0.1:  # 0.1 radian threshold
            cmd.angular.z = yaw_diff * 0.5  # Proportional control

        self.motion_command_publisher.publish(cmd)

        # Update progress
        max_distance = params.get('max_distance', 10.0)  # Expected max distance
        self.current_behavior.progress = min(1.0, (max_distance - distance) / max_distance)

    def execute_manipulation_behavior(self):
        """Execute manipulation behavior"""
        params = self.current_behavior.parameters
        target_object = params.get('target_object', 'object')

        # For simulation, we'll just update progress
        elapsed = time.time() - self.current_behavior.start_time
        expected_duration = params.get('duration', 5.0)
        self.current_behavior.progress = min(1.0, elapsed / expected_duration)

        # Publish joint commands for manipulation
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'

        # Simulate manipulation actions
        joint_cmd.name = ['gripper_joint', 'arm_joint_1', 'arm_joint_2']
        joint_cmd.position = [0.0, 0.5, 0.3]  # Example positions
        joint_cmd.velocity = [0.1, 0.1, 0.1]
        joint_cmd.effort = [10.0, 5.0, 5.0]

        self.joint_command_publisher.publish(joint_cmd)

        # Complete after expected duration
        if elapsed >= expected_duration:
            self.complete_current_behavior()

    def execute_interaction_behavior(self):
        """Execute interaction behavior"""
        params = self.current_behavior.parameters
        target_entity = params.get('target_entity', 'human')

        # Update progress based on interaction simulation
        elapsed = time.time() - self.current_behavior.start_time
        expected_duration = params.get('duration', 3.0)
        self.current_behavior.progress = min(1.0, elapsed / expected_duration)

        # For interaction, we might need to orient toward the target
        # This would involve more complex behavior in a real system

        # Complete after duration
        if elapsed >= expected_duration:
            self.complete_current_behavior()

    def execute_speaking_behavior(self):
        """Execute speaking behavior"""
        params = self.current_behavior.parameters
        text = params.get('text', 'Hello')

        # Update progress
        elapsed = time.time() - self.current_behavior.start_time
        expected_duration = len(text.split()) * 0.5  # Rough estimate: 0.5s per word
        self.current_behavior.progress = min(1.0, elapsed / expected_duration)

        # In a real system, this would trigger text-to-speech
        # For simulation, just log the text
        if elapsed < 1.0:  # Log once at the start
            self.get_logger().info(f"Speaking: {text}")

        # Complete after duration
        if elapsed >= expected_duration:
            self.complete_current_behavior()

    def execute_waiting_behavior(self):
        """Execute waiting behavior"""
        params = self.current_behavior.parameters
        duration = params.get('duration', 2.0)

        # Update progress
        elapsed = time.time() - self.current_behavior.start_time
        self.current_behavior.progress = min(1.0, elapsed / duration)

        # Stop any motion
        stop_cmd = Twist()
        self.motion_command_publisher.publish(stop_cmd)

        # Complete after duration
        if elapsed >= duration:
            self.complete_current_behavior()

    def execute_emergency_stop_behavior(self):
        """Execute emergency stop behavior"""
        # Ensure all motion is stopped
        stop_cmd = Twist()
        self.motion_command_publisher.publish(stop_cmd)

        # Keep this behavior active until cleared by safety system
        self.current_behavior.progress = 1.0  # Mark as complete internally but keep active

    def execute_safety_response_behavior(self):
        """Execute safety response behavior"""
        # Stop motion
        stop_cmd = Twist()
        self.motion_command_publisher.publish(stop_cmd)

        # Check if safety conditions are resolved
        if not self.safety_emergency_active:
            self.complete_current_behavior()

        self.current_behavior.progress = 0.5  # Safety response is ongoing

    def execute_idle_behavior(self):
        """Execute idle behavior"""
        # In idle state, just maintain position
        stop_cmd = Twist()
        self.motion_command_publisher.publish(stop_cmd)

        # Update progress (idle is always "complete")
        self.current_behavior.progress = 1.0

    def complete_current_behavior(self):
        """Complete the current behavior"""
        if not self.current_behavior:
            return

        # Log completion
        self.get_logger().info(f"Completed behavior: {self.current_behavior.behavior.value}")

        # Update status
        self.current_behavior.status = 'completed'
        self.behavior_history.append(self.current_behavior)

        # Publish completion
        cmd_msg = String()
        cmd_msg.data = f"BEHAVIOR_COMPLETE: {self.current_behavior.behavior.value}"
        self.behavior_command_publisher.publish(cmd_msg)

        # Clear current behavior
        self.current_behavior = None

        # Check for next behavior in queue
        self.process_behavior_queue()

    def process_behavior_queue(self):
        """Process behaviors in the queue"""
        # For now, just transition to idle when no behavior is active
        if not self.current_behavior:
            idle_behavior = ActiveBehavior(
                id=f"idle_{int(time.time())}",
                behavior=BehaviorState.IDLE,
                priority=BehaviorPriority.LOW,
                parameters={},
                start_time=time.time(),
                progress=1.0,
                status='active'
            )
            self.current_behavior = idle_behavior

    def check_behavior_timeouts(self):
        """Check for behavior timeouts"""
        if self.current_behavior:
            elapsed = time.time() - self.current_behavior.start_time
            if elapsed > self.behavior_timeout:
                self.get_logger().warn(f"Behavior {self.current_behavior.behavior.value} timed out")
                self.current_behavior.status = 'timed_out'
                self.behavior_history.append(self.current_behavior)
                self.current_behavior = None

    def update_behavior_history(self):
        """Update behavior history"""
        # Keep only recent history
        if len(self.behavior_history) > 50:
            self.behavior_history = self.behavior_history[-50:]

    def publish_behavior_status(self):
        """Publish behavior status"""
        status_parts = []

        if self.current_behavior:
            status_parts.append(f"Current: {self.current_behavior.behavior.value}")
            status_parts.append(f"Progress: {self.current_behavior.progress:.2f}")
            status_parts.append(f"Priority: {self.current_behavior.priority.name}")
        else:
            status_parts.append("Current: None")

        status_parts.append(f"Queue: {len(self.behavior_requests)}")
        status_parts.append(f"History: {len(self.behavior_history)}")
        status_parts.append(f"Emergency: {'ACTIVE' if self.safety_emergency_active else 'INACTIVE'}")

        status_msg = String()
        status_msg.data = f"BEHAVIOR: {', '.join(status_parts)}"
        self.behavior_status_publisher.publish(status_msg)

    def get_current_behavior_info(self) -> Optional[Dict[str, any]]:
        """Get information about the current behavior"""
        if not self.current_behavior:
            return None

        return {
            'behavior': self.current_behavior.behavior.value,
            'priority': self.current_behavior.priority.name,
            'progress': self.current_behavior.progress,
            'status': self.current_behavior.status,
            'parameters': self.current_behavior.parameters,
            'start_time': self.current_behavior.start_time
        }

    def get_behavior_stats(self) -> Dict[str, any]:
        """Get behavior statistics"""
        return {
            'current_behavior': self.current_behavior.behavior.value if self.current_behavior else 'none',
            'request_queue_size': len(self.behavior_requests),
            'behavior_history_size': len(self.behavior_history),
            'safety_emergency_active': self.safety_emergency_active,
            'enabled': self.behavior_coordination_enabled
        }

    def trigger_behavior_transition(self, new_behavior: BehaviorState):
        """Force a behavior transition"""
        if not self.is_valid_transition(self.current_behavior.behavior if self.current_behavior else BehaviorState.IDLE, new_behavior):
            self.get_logger().warn(f"Invalid forced transition: {self.current_behavior.behavior.value if self.current_behavior else 'IDLE'} -> {new_behavior.value}")
            return

        # Add current behavior to history if exists
        if self.current_behavior:
            self.current_behavior.status = 'interrupted'
            self.behavior_history.append(self.current_behavior)

        # Create new behavior
        new_active = ActiveBehavior(
            id=f"forced_{int(time.time())}",
            behavior=new_behavior,
            priority=BehaviorPriority.HIGH,
            parameters={},
            start_time=time.time(),
            progress=0.0,
            status='active'
        )

        self.current_behavior = new_active

        self.get_logger().info(f"Force transitioned to: {new_behavior.value}")

    def determine_behavior_from_command(self, command: str) -> BehaviorState:
        """Determine behavior from a simple command string"""
        command_lower = command.lower()

        if any(word in command_lower for word in ['go', 'move', 'navigate', 'walk', 'drive']):
            return BehaviorState.NAVIGATING
        elif any(word in command_lower for word in ['grasp', 'pick', 'take', 'grab', 'lift', 'place', 'put']):
            return BehaviorState.MANIPULATING
        elif any(word in command_lower for word in ['speak', 'say', 'tell', 'hello', 'hi']):
            return BehaviorState.SPEAKING
        elif any(word in command_lower for word in ['look', 'point', 'wave', 'greet', 'interact']):
            return BehaviorState.INTERACTING
        elif any(word in command_lower for word in ['wait', 'stop', 'pause']):
            return BehaviorState.WAITING
        else:
            return BehaviorState.IDLE

    def get_current_yaw(self) -> float:
        """Get the current yaw angle from the robot's orientation"""
        # Simplified: extract yaw from quaternion
        q = self.current_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def enable_behavior_coordination(self, enable: bool):
        """Enable or disable behavior coordination"""
        self.behavior_coordination_enabled = enable
        self.get_logger().info(f"Behavior coordination {'enabled' if enable else 'disabled'}")


def main(args=None):
    rclpy.init(args=args)

    behavior_coordinator_node = BehaviorCoordinatorNode()

    try:
        rclpy.spin(behavior_coordinator_node)
    except KeyboardInterrupt:
        pass
    finally:
        behavior_coordinator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()