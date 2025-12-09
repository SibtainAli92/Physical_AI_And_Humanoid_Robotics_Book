#!/usr/bin/env python3

"""
Decision Maker Node for Humanoid Robot AI Brain

This node handles AI decision making, planning, and high-level reasoning
using techniques similar to those in NVIDIA Isaac.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from geometry_msgs.msg import Pose, PoseStamped, Twist, Vector3
from sensor_msgs.msg import JointState
from nav_msgs.msg import Path, Odometry
from ai_brain_isaac.msg import Decision, DecisionRequest, DecisionResponse
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from enum import Enum
from dataclasses import dataclass


class DecisionType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    EXPLORATION = "exploration"
    SAFETY = "safety"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task for the robot to perform"""
    id: str
    type: DecisionType
    priority: TaskPriority
    goal: Pose
    constraints: Dict[str, float]
    created_time: float
    deadline: float
    status: str  # 'pending', 'in_progress', 'completed', 'failed'


class DecisionMakerNode(Node):
    def __init__(self):
        super().__init__('decision_maker_node')

        # Publishers for decisions and actions
        self.decision_publisher = self.create_publisher(Decision, 'ai/decision', 10)
        self.motion_command_publisher = self.create_publisher(Twist, 'motion_commands', 10)
        self.path_publisher = self.create_publisher(Path, 'ai/planned_path', 10)
        self.decision_status_publisher = self.create_publisher(String, 'ai/decision_status', 10)

        # Subscribers for environment and state data
        self.perception_subscriber = self.create_subscription(
            String, 'ai/perception_status', self.perception_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.task_request_subscriber = self.create_subscription(
            DecisionRequest, 'ai/task_request', self.task_request_callback, 10)

        # Timer for decision making
        self.decision_timer = self.create_timer(0.2, self.decision_callback)  # 5 Hz

        # Decision making state
        self.current_pose = Pose()
        self.current_joint_states = {}
        self.perception_data = {}
        self.task_queue = []
        self.current_task: Optional[Task] = None
        self.decision_enabled = True
        self.sim_time = time.time()
        self.last_decision_time = time.time()

        # Decision making parameters
        self.safety_threshold = 0.5
        self.planning_horizon = 5.0  # seconds
        self.replan_threshold = 0.3  # distance threshold for replanning

        # Path planning
        self.current_path = []
        self.path_index = 0

        # Threading lock for state access
        self.state_lock = threading.Lock()

        self.get_logger().info('Decision Maker Node initialized')

    def perception_callback(self, msg):
        """Callback for perception data"""
        with self.state_lock:
            # Parse perception status message for relevant information
            self.perception_data = {
                'timestamp': time.time(),
                'status': msg.data
            }

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.state_lock:
            self.current_pose = msg.pose.pose

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        with self.state_lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_joint_states[name] = msg.position[i]

    def task_request_callback(self, msg):
        """Callback for task requests"""
        with self.state_lock:
            # Create a new task from the request
            new_task = Task(
                id=f"task_{len(self.task_queue)}",
                type=DecisionType(msg.task_type),
                priority=TaskPriority(msg.priority),
                goal=msg.goal,
                constraints=msg.constraints,
                created_time=time.time(),
                deadline=time.time() + msg.timeout,
                status='pending'
            )

            # Insert task based on priority
            self.insert_task_by_priority(new_task)
            self.get_logger().info(f'Received task request: {new_task.id}, type: {new_task.type.value}')

    def insert_task_by_priority(self, task: Task):
        """Insert task into queue based on priority"""
        # Find the correct position based on priority (higher priority first)
        inserted = False
        for i, existing_task in enumerate(self.task_queue):
            if task.priority.value > existing_task.priority.value:
                self.task_queue.insert(i, task)
                inserted = True
                break

        if not inserted:
            self.task_queue.append(task)

    def decision_callback(self):
        """Main decision making callback"""
        if not self.decision_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_decision_time
        self.last_decision_time = current_time

        with self.state_lock:
            # Process tasks in queue
            self.process_task_queue()

            # Make decisions based on current state and tasks
            decision = self.make_decision()

            if decision:
                # Publish decision
                decision_msg = Decision()
                decision_msg.header.stamp = self.get_clock().now().to_msg()
                decision_msg.header.frame_id = 'decision_frame'
                decision_msg.type = decision['type']
                decision_msg.action = decision['action']
                decision_msg.confidence = decision['confidence']
                decision_msg.execution_time = current_time

                self.decision_publisher.publish(decision_msg)

                # Execute decision (for now, just log it)
                self.execute_decision(decision)

        # Publish decision status
        status_msg = String()
        status_msg.data = f"DECISION: Tasks={len(self.task_queue)}, Current={self.current_task.id if self.current_task else 'None'}, Time={current_time:.2f}"
        self.decision_status_publisher.publish(status_msg)

    def process_task_queue(self):
        """Process tasks in the queue"""
        if not self.task_queue:
            return

        # Check if current task is completed or failed
        if self.current_task:
            if self.is_task_completed(self.current_task):
                self.current_task.status = 'completed'
                self.current_task = None
            elif current_time > self.current_task.deadline:
                self.current_task.status = 'failed'
                self.current_task = None

        # If no current task, get the highest priority task
        if not self.current_task and self.task_queue:
            self.current_task = self.task_queue.pop(0)
            self.current_task.status = 'in_progress'
            self.get_logger().info(f'Starting task: {self.current_task.id}')

    def is_task_completed(self, task: Task) -> bool:
        """Check if a task is completed"""
        if task.type == DecisionType.NAVIGATION:
            # Check if we're close to the goal
            distance_to_goal = self.calculate_distance_to_pose(task.goal)
            return distance_to_goal < 0.2  # 20cm threshold
        elif task.type == DecisionType.INTERACTION:
            # For interaction tasks, completion might be based on other factors
            # This is a simplified check
            return False
        else:
            # For other task types, use a default completion check
            return False

    def calculate_distance_to_pose(self, goal_pose: Pose) -> float:
        """Calculate distance from current pose to goal pose"""
        dx = goal_pose.position.x - self.current_pose.position.x
        dy = goal_pose.position.y - self.current_pose.position.y
        dz = goal_pose.position.z - self.current_pose.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def make_decision(self) -> Optional[Dict]:
        """Make a decision based on current state and tasks"""
        if not self.current_task:
            # If no specific task, decide on general behavior
            return self.make_general_decision()

        # Make decision based on current task
        if self.current_task.type == DecisionType.NAVIGATION:
            return self.make_navigation_decision()
        elif self.current_task.type == DecisionType.MANIPULATION:
            return self.make_manipulation_decision()
        elif self.current_task.type == DecisionType.INTERACTION:
            return self.make_interaction_decision()
        elif self.current_task.type == DecisionType.SAFETY:
            return self.make_safety_decision()
        else:
            return self.make_general_decision()

    def make_general_decision(self) -> Dict:
        """Make a general decision when no specific task is active"""
        # For now, return a simple decision to maintain current behavior
        return {
            'type': 'general',
            'action': 'maintain_current_behavior',
            'confidence': 1.0
        }

    def make_navigation_decision(self) -> Dict:
        """Make a navigation decision"""
        if not self.current_task:
            return None

        # Calculate path to goal
        path = self.plan_path_to_goal(self.current_task.goal)

        if path:
            # Follow the path
            next_waypoint = path[0] if path else self.current_task.goal
            direction = self.calculate_direction_to_pose(next_waypoint)

            # Create motion command
            cmd = Twist()
            cmd.linear.x = direction[0] * 0.3  # Move at 0.3 m/s
            cmd.linear.y = direction[1] * 0.3
            cmd.angular.z = direction[2] * 0.5  # Turn if needed

            self.motion_command_publisher.publish(cmd)

            return {
                'type': 'navigation',
                'action': 'move_to_goal',
                'confidence': 0.9,
                'target': [next_waypoint.position.x, next_waypoint.position.y]
            }
        else:
            # Can't find path, need to replan or request help
            return {
                'type': 'navigation',
                'action': 'path_unavailable',
                'confidence': 0.3
            }

    def make_manipulation_decision(self) -> Dict:
        """Make a manipulation decision"""
        # For now, return a simple manipulation decision
        return {
            'type': 'manipulation',
            'action': 'manipulation_not_implemented',
            'confidence': 0.5
        }

    def make_interaction_decision(self) -> Dict:
        """Make an interaction decision"""
        # For now, return a simple interaction decision
        return {
            'type': 'interaction',
            'action': 'interaction_not_implemented',
            'confidence': 0.5
        }

    def make_safety_decision(self) -> Dict:
        """Make a safety decision"""
        # Check for safety concerns in perception data
        if 'COLLISION' in self.perception_data.get('status', ''):
            # Emergency stop
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            self.motion_command_publisher.publish(cmd)

            return {
                'type': 'safety',
                'action': 'emergency_stop',
                'confidence': 1.0
            }

        return {
            'type': 'safety',
            'action': 'all_clear',
            'confidence': 0.9
        }

    def plan_path_to_goal(self, goal_pose: Pose) -> List[Pose]:
        """Plan a path to the goal pose (simplified implementation)"""
        # This is a very simplified path planning implementation
        # In a real system, this would use sophisticated algorithms like A*, RRT, etc.

        # For now, create a straight line path to the goal
        path = []

        # Calculate the direction vector to the goal
        dx = goal_pose.position.x - self.current_pose.position.x
        dy = goal_pose.position.y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0.1:  # Only plan if goal is far enough
            # Create intermediate waypoints
            num_waypoints = max(2, int(distance / 0.2))  # Waypoints every 20cm

            for i in range(1, num_waypoints + 1):
                fraction = i / num_waypoints
                waypoint = Pose()
                waypoint.position.x = self.current_pose.position.x + dx * fraction
                waypoint.position.y = self.current_pose.position.y + dy * fraction
                waypoint.position.z = self.current_pose.position.z  # Maintain current height
                path.append(waypoint)

        return path

    def calculate_direction_to_pose(self, target_pose: Pose) -> Tuple[float, float, float]:
        """Calculate normalized direction vector to target pose"""
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0:
            return (dx/distance, dy/distance, 0.0)  # 0.0 for angular component for now
        else:
            return (0.0, 0.0, 0.0)

    def execute_decision(self, decision: Dict):
        """Execute the decision"""
        # In this simplified implementation, decisions are already executed
        # when they're made (e.g., motion commands are published immediately)
        pass

    def enable_decision_making(self, enable: bool):
        """Enable or disable decision making"""
        self.decision_enabled = enable
        self.get_logger().info(f"Decision making {'enabled' if enable else 'disabled'}")

    def add_task(self, task_type: DecisionType, goal: Pose, priority: TaskPriority = TaskPriority.MEDIUM, timeout: float = 10.0):
        """Add a task to the task queue"""
        with self.state_lock:
            new_task = Task(
                id=f"task_{len(self.task_queue)}",
                type=task_type,
                priority=priority,
                goal=goal,
                constraints={},
                created_time=time.time(),
                deadline=time.time() + timeout,
                status='pending'
            )
            self.insert_task_by_priority(new_task)
            self.get_logger().info(f'Added task: {new_task.id}, type: {task_type.value}')

    def get_decision_stats(self) -> Dict[str, float]:
        """Get decision making statistics"""
        return {
            'task_queue_size': len(self.task_queue),
            'current_task': self.current_task.id if self.current_task else None,
            'last_decision_time': self.last_decision_time,
            'enabled': self.decision_enabled
        }

    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs"""
        with self.state_lock:
            return [task.id for task in self.task_queue] + ([self.current_task.id] if self.current_task else [])


def main(args=None):
    rclpy.init(args=args)

    decision_maker_node = DecisionMakerNode()

    try:
        rclpy.spin(decision_maker_node)
    except KeyboardInterrupt:
        pass
    finally:
        decision_maker_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()