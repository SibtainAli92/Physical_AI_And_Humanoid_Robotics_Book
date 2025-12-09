#!/usr/bin/env python3

"""
Cognitive Controller Node for Humanoid Robot AI Brain

This node serves as the central coordinator for the AI Brain system,
managing the interaction between perception, decision making, learning,
and memory components similar to higher-level cognitive functions.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose, Vector3
from nav_msgs.msg import Odometry
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum


class CognitiveState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    DECIDING = "deciding"
    ACTING = "acting"
    LEARNING = "learning"
    MEMORY_PROCESSING = "memory_processing"


@dataclass
class CognitiveTask:
    """Represents a cognitive task for the AI brain"""
    id: str
    state: CognitiveState
    priority: int
    created_time: float
    deadline: float
    description: str
    dependencies: List[str]


class CognitiveControllerNode(Node):
    def __init__(self):
        super().__init__('cognitive_controller_node')

        # Publishers for cognitive system status
        self.cognitive_status_publisher = self.create_publisher(String, 'ai/cognitive_status', 10)
        self.system_command_publisher = self.create_publisher(String, 'ai/system_command', 10)
        self.attention_publisher = self.create_publisher(Float32, 'ai/attention', 10)

        # Subscribers for cognitive component status
        self.perception_status_subscriber = self.create_subscription(
            String, 'ai/perception_status', self.perception_status_callback, 10)
        self.decision_status_subscriber = self.create_subscription(
            String, 'ai/decision_status', self.decision_status_callback, 10)
        self.learning_status_subscriber = self.create_subscription(
            String, 'ai/learning_status', self.learning_status_callback, 10)
        self.memory_status_subscriber = self.create_subscription(
            String, 'ai/memory_status', self.memory_status_callback, 10)

        # Subscribers for sensor and state data
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Timer for cognitive control
        self.cognitive_timer = self.create_timer(0.1, self.cognitive_control_callback)  # 10 Hz

        # Cognitive system state
        self.current_cognitive_state = CognitiveState.IDLE
        self.previous_cognitive_state = CognitiveState.IDLE
        self.cognitive_tasks = []
        self.active_task: Optional[CognitiveTask] = None
        self.cognitive_enabled = True
        self.attention_level = 1.0  # 0.0 to 1.0
        self.focus_threshold = 0.7  # Minimum attention for focused processing

        # Component status tracking
        self.perception_status = "idle"
        self.decision_status = "idle"
        self.learning_status = "idle"
        self.memory_status = "idle"

        # System state
        self.current_pose = Pose()
        self.current_joint_states = {}
        self.balance_metrics = {'tilt_angle': 0.0, 'com_distance': 0.0, 'stability': 1.0}
        self.threat_level = 0.0  # 0.0 to 1.0
        self.opportunity_level = 0.0  # 0.0 to 1.0

        # Cognitive parameters
        self.cognitive_cycle_time = 0.1  # 100ms per cognitive cycle
        self.state_transition_threshold = 0.5
        self.max_cognitive_load = 10  # Maximum concurrent cognitive tasks

        # Threading lock for state access
        self.state_lock = threading.Lock()

        self.get_logger().info('Cognitive Controller Node initialized')

    def perception_status_callback(self, msg):
        """Callback for perception system status"""
        with self.state_lock:
            self.perception_status = msg.data

    def decision_status_callback(self, msg):
        """Callback for decision system status"""
        with self.state_lock:
            self.decision_status = msg.data

    def learning_status_callback(self, msg):
        """Callback for learning system status"""
        with self.state_lock:
            self.learning_status = msg.data

    def memory_status_callback(self, msg):
        """Callback for memory system status"""
        with self.state_lock:
            self.memory_status = msg.data

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        with self.state_lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_joint_states[name] = msg.position[i]

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.state_lock:
            self.current_pose = msg.pose.pose

    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.state_lock:
            # Calculate balance metrics from IMU data
            self.balance_metrics['tilt_angle'] = math.sqrt(
                msg.orientation.x**2 + msg.orientation.y**2
            )
            # Simplified stability calculation
            self.balance_metrics['stability'] = max(0.0, 1.0 - self.balance_metrics['tilt_angle'])

    def cognitive_control_callback(self):
        """Main cognitive control callback"""
        if not self.cognitive_enabled:
            return

        current_time = time.time()

        with self.state_lock:
            # Update system state
            self.update_system_state()

            # Determine cognitive state based on system state
            new_cognitive_state = self.determine_cognitive_state()

            # Transition to new cognitive state if needed
            if new_cognitive_state != self.current_cognitive_state:
                self.transition_cognitive_state(new_cognitive_state)

            # Execute current cognitive state
            self.execute_cognitive_state()

            # Process cognitive tasks
            self.process_cognitive_tasks()

            # Update attention level based on system needs
            self.update_attention_level()

        # Publish cognitive status
        status_msg = String()
        status_msg.data = f"COGNITIVE: State={self.current_cognitive_state.value}, Tasks={len(self.cognitive_tasks)}, Attention={self.attention_level:.2f}"
        self.cognitive_status_publisher.publish(status_msg)

        # Publish attention level
        attention_msg = Float32()
        attention_msg.data = self.attention_level
        self.attention_publisher.publish(attention_msg)

    def update_system_state(self):
        """Update internal system state based on sensor data"""
        # Update threat level based on sensor data
        self.threat_level = self.assess_threat_level()

        # Update opportunity level based on environment
        self.opportunity_level = self.assess_opportunity_level()

    def assess_threat_level(self) -> float:
        """Assess current threat level based on sensor data"""
        threat = 0.0

        # Check balance metrics
        if self.balance_metrics['stability'] < 0.3:
            threat += 0.5  # Unstable balance is a threat

        # Check for close obstacles (simplified - would use laser scan data in real system)
        # For now, we'll simulate based on balance
        if self.balance_metrics['tilt_angle'] > 0.5:
            threat += 0.3

        # Cap threat level
        return min(1.0, threat)

    def assess_opportunity_level(self) -> float:
        """Assess current opportunity level based on environment"""
        opportunity = 0.0

        # Simple opportunity assessment
        # In a real system, this would check for goals, tasks, or learning opportunities
        if len(self.current_joint_states) > 0:  # If we have joint data
            opportunity += 0.1

        # Cap opportunity level
        return min(1.0, opportunity)

    def determine_cognitive_state(self) -> CognitiveState:
        """Determine the appropriate cognitive state based on system state"""
        # Priority-based state determination
        if self.threat_level > 0.8:
            return CognitiveState.DECIDING  # High threat requires immediate decision
        elif self.threat_level > 0.5:
            return CognitiveState.PERCEIVING  # Moderate threat requires perception
        elif self.attention_level < self.focus_threshold:
            return CognitiveState.IDLE  # Low attention means idle
        elif self.opportunity_level > 0.7:
            return CognitiveState.LEARNING  # High opportunity for learning
        elif self.active_task:
            return self.active_task.state
        else:
            return CognitiveState.IDLE

    def transition_cognitive_state(self, new_state: CognitiveState):
        """Handle transition to a new cognitive state"""
        self.get_logger().debug(f'Transitioning from {self.current_cognitive_state.value} to {new_state.value}')

        # Execute exit procedures for current state
        self.exit_cognitive_state(self.current_cognitive_state)

        # Store previous state
        self.previous_cognitive_state = self.current_cognitive_state
        self.current_cognitive_state = new_state

        # Execute entry procedures for new state
        self.enter_cognitive_state(new_state)

    def exit_cognitive_state(self, state: CognitiveState):
        """Execute exit procedures for a cognitive state"""
        if state == CognitiveState.ACTING:
            # When exiting acting state, ensure safe transition
            stop_cmd = Twist()
            self.system_command_publisher.publish(String(data=f"motion_command:{stop_cmd.linear.x},{stop_cmd.linear.y},{stop_cmd.angular.z}"))

    def enter_cognitive_state(self, state: CognitiveState):
        """Execute entry procedures for a cognitive state"""
        if state == CognitiveState.PERCEIVING:
            # Increase attention for perception
            self.attention_level = min(1.0, self.attention_level + 0.2)
        elif state == CognitiveState.REASONING:
            # Focus attention for reasoning
            self.attention_level = min(1.0, self.attention_level + 0.1)
        elif state == CognitiveState.DECIDING:
            # Maximum attention for decision making
            self.attention_level = 1.0

    def execute_cognitive_state(self):
        """Execute the current cognitive state"""
        if self.current_cognitive_state == CognitiveState.IDLE:
            self.execute_idle_state()
        elif self.current_cognitive_state == CognitiveState.PERCEIVING:
            self.execute_perceiving_state()
        elif self.current_cognitive_state == CognitiveState.REASONING:
            self.execute_reasoning_state()
        elif self.current_cognitive_state == CognitiveState.DECIDING:
            self.execute_deciding_state()
        elif self.current_cognitive_state == CognitiveState.ACTING:
            self.execute_acting_state()
        elif self.current_cognitive_state == CognitiveState.LEARNING:
            self.execute_learning_state()
        elif self.current_cognitive_state == CognitiveState.MEMORY_PROCESSING:
            self.execute_memory_processing_state()

    def execute_idle_state(self):
        """Execute idle cognitive state"""
        # In idle state, maintain basic awareness
        pass

    def execute_perceiving_state(self):
        """Execute perceiving cognitive state"""
        # In perceiving state, focus on sensor data processing
        self.system_command_publisher.publish(String(data="perception:enable"))

    def execute_reasoning_state(self):
        """Execute reasoning cognitive state"""
        # In reasoning state, process information and draw conclusions
        pass

    def execute_deciding_state(self):
        """Execute deciding cognitive state"""
        # In deciding state, make critical decisions
        if self.threat_level > 0.5:
            self.system_command_publisher.publish(String(data="safety_protocol:activate"))

    def execute_acting_state(self):
        """Execute acting cognitive state"""
        # In acting state, execute motor commands
        pass

    def execute_learning_state(self):
        """Execute learning cognitive state"""
        # In learning state, focus on experience and adaptation
        self.system_command_publisher.publish(String(data="learning:enable"))

    def execute_memory_processing_state(self):
        """Execute memory processing cognitive state"""
        # In memory processing state, consolidate and organize memories
        self.system_command_publisher.publish(String(data="memory:consolidate"))

    def process_cognitive_tasks(self):
        """Process cognitive tasks in the queue"""
        if len(self.cognitive_tasks) > self.max_cognitive_load:
            # Remove lowest priority tasks
            self.cognitive_tasks.sort(key=lambda x: x.priority, reverse=True)
            self.cognitive_tasks = self.cognitive_tasks[:self.max_cognitive_load]

        # Process tasks based on priority
        for task in self.cognitive_tasks[:]:  # Use slice to avoid modification during iteration
            if time.time() > task.deadline:
                self.cognitive_tasks.remove(task)
                self.get_logger().info(f'Task {task.id} expired')

    def update_attention_level(self):
        """Update attention level based on system needs"""
        base_attention = 0.5  # Base attention level

        # Increase attention based on threat level
        attention_from_threat = self.threat_level * 0.3

        # Increase attention based on opportunity level
        attention_from_opportunity = self.opportunity_level * 0.2

        # Calculate new attention level
        new_attention = base_attention + attention_from_threat + attention_from_opportunity
        self.attention_level = max(0.1, min(1.0, new_attention))  # Clamp between 0.1 and 1.0

    def add_cognitive_task(self, state: CognitiveState, description: str, priority: int = 1, timeout: float = 10.0):
        """Add a cognitive task to the queue"""
        with self.state_lock:
            task = CognitiveTask(
                id=f"task_{len(self.cognitive_tasks)}",
                state=state,
                priority=priority,
                created_time=time.time(),
                deadline=time.time() + timeout,
                description=description,
                dependencies=[]
            )
            self.cognitive_tasks.append(task)

            # Sort tasks by priority (highest first)
            self.cognitive_tasks.sort(key=lambda x: x.priority, reverse=True)

            self.get_logger().info(f'Added cognitive task: {task.id}, state: {state.value}, priority: {priority}')

    def enable_cognitive_system(self, enable: bool):
        """Enable or disable the cognitive system"""
        self.cognitive_enabled = enable
        self.get_logger().info(f"Cognitive system {'enabled' if enable else 'disabled'}")

    def request_cognitive_state_change(self, state: CognitiveState):
        """Request a specific cognitive state change"""
        with self.state_lock:
            if self.current_cognitive_state != state:
                self.transition_cognitive_state(state)

    def get_cognitive_stats(self) -> Dict[str, any]:
        """Get cognitive system statistics"""
        return {
            'current_state': self.current_cognitive_state.value,
            'previous_state': self.previous_cognitive_state.value,
            'task_queue_size': len(self.cognitive_tasks),
            'attention_level': self.attention_level,
            'threat_level': self.threat_level,
            'opportunity_level': self.opportunity_level,
            'enabled': self.cognitive_enabled,
            'balance_stability': self.balance_metrics['stability']
        }

    def get_system_assessment(self) -> Dict[str, float]:
        """Get overall system assessment"""
        return {
            'threat_level': self.threat_level,
            'opportunity_level': self.opportunity_level,
            'attention_level': self.attention_level,
            'balance_stability': self.balance_metrics['stability'],
            'cognitive_load': len(self.cognitive_tasks) / self.max_cognitive_load
        }


def main(args=None):
    rclpy.init(args=args)

    cognitive_controller_node = CognitiveControllerNode()

    try:
        rclpy.spin(cognitive_controller_node)
    except KeyboardInterrupt:
        pass
    finally:
        cognitive_controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()