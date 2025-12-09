#!/usr/bin/env python3

"""
Task Execution Node for Humanoid Robot VLA System

This node executes tasks planned by the action planner, coordinating
the robot's movements and actions to complete requested tasks.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from geometry_msgs.msg import Twist, Pose, Point, Vector3
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry, Path
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import json


class ExecutionState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    SPEAKING = "speaking"


@dataclass
class TaskStep:
    """Represents a single step in a task"""
    id: str
    action: str  # 'move', 'grasp', 'speak', etc.
    parameters: Dict[str, any]
    duration: float  # Expected duration in seconds
    dependencies: List[str]  # IDs of steps that must complete first
    priority: int


@dataclass
class Task:
    """Represents a complete task to be executed"""
    id: str
    type: TaskType
    steps: List[TaskStep]
    created_time: float
    deadline: float
    status: ExecutionState
    current_step_index: int
    progress: float  # 0.0 to 1.0


class TaskExecutionNode(Node):
    def __init__(self):
        super().__init__('task_execution_node')

        # Publishers for task execution
        self.task_status_publisher = self.create_publisher(String, 'vla/task_status', 10)
        self.motion_command_publisher = self.create_publisher(Twist, 'motion_commands', 10)
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.execution_status_publisher = self.create_publisher(String, 'vla/execution_status', 10)
        self.feedback_publisher = self.create_publisher(String, 'vla/execution_feedback', 10)

        # Subscribers for task inputs
        self.task_plan_subscriber = self.create_subscription(
            String, 'vla/task_plan', self.task_plan_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.task_cancel_subscriber = self.create_subscription(
            String, 'vla/task_cancel', self.task_cancel_callback, 10)

        # Timer for task execution
        self.execution_timer = self.create_timer(0.1, self.task_execution_callback)  # 10 Hz

        # Task execution state
        self.current_task: Optional[Task] = None
        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.task_execution_enabled = True
        self.sim_time = time.time()
        self.last_execution_time = time.time()

        # Robot state
        self.current_pose = Pose()
        self.current_joint_states = {}
        self.current_imu = None
        self.current_twist = Twist()
        self.balance_stable = True

        # Execution parameters
        self.navigation_speed = 0.3  # m/s
        self.rotation_speed = 0.5  # rad/s
        self.manipulation_speed = 0.1  # m/s for manipulation
        self.safety_margin = 0.3  # meters for obstacle avoidance
        self.execution_timeout = 30.0  # seconds before task timeout
        self.step_timeout = 10.0  # seconds before step timeout

        # Task execution tracking
        self.current_step_start_time = 0.0
        self.step_execution_timeout = False

        # Threading lock for data access
        self.data_lock = threading.Lock()

        self.get_logger().info('Task Execution Node initialized')

    def task_plan_callback(self, msg):
        """Callback for task plans"""
        with self.data_lock:
            try:
                # Parse the task plan from JSON
                task_data = json.loads(msg.data)

                # Create a new task from the plan
                task_steps = []
                for step_data in task_data.get('steps', []):
                    step = TaskStep(
                        id=step_data['id'],
                        action=step_data['action'],
                        parameters=step_data.get('parameters', {}),
                        duration=step_data.get('duration', 2.0),
                        dependencies=step_data.get('dependencies', []),
                        priority=step_data.get('priority', 1)
                    )
                    task_steps.append(step)

                task = Task(
                    id=task_data['id'],
                    type=TaskType(task_data.get('type', 'navigation')),
                    steps=task_steps,
                    created_time=time.time(),
                    deadline=time.time() + task_data.get('timeout', 60.0),
                    status=ExecutionState.IDLE,
                    current_step_index=0,
                    progress=0.0
                )

                # Add to queue or execute immediately if no current task
                if not self.current_task:
                    self.start_task_execution(task)
                else:
                    self.task_queue.append(task)
                    self.get_logger().info(f'Added task {task.id} to queue, queue size: {len(self.task_queue)}')

            except json.JSONDecodeError:
                # If not JSON, try to parse as simple string command
                self.create_simple_task(msg.data)

    def create_simple_task(self, command: str):
        """Create a simple task from a string command"""
        # Determine task type from command
        if any(word in command.lower() for word in ['go', 'move', 'navigate', 'walk']):
            task_type = TaskType.NAVIGATION
            action = 'move_to'
        elif any(word in command.lower() for word in ['grasp', 'pick', 'take', 'grab']):
            task_type = TaskType.MANIPULATION
            action = 'grasp'
        elif any(word in command.lower() for word in ['speak', 'say', 'tell']):
            task_type = TaskType.SPEAKING
            action = 'speak'
        else:
            task_type = TaskType.INTERACTION
            action = 'wait'

        # Create a simple task with one step
        step = TaskStep(
            id=f"step_{int(time.time())}",
            action=action,
            parameters={'command': command},
            duration=5.0,
            dependencies=[],
            priority=1
        )

        task = Task(
            id=f"simple_task_{int(time.time())}",
            type=task_type,
            steps=[step],
            created_time=time.time(),
            deadline=time.time() + 30.0,
            status=ExecutionState.IDLE,
            current_step_index=0,
            progress=0.0
        )

        # Add to queue or execute immediately
        if not self.current_task:
            self.start_task_execution(task)
        else:
            self.task_queue.append(task)

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.data_lock:
            self.current_pose = msg.pose.pose
            self.current_twist = msg.twist.twist

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        with self.data_lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_joint_states[name] = msg.position[i]

    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.data_lock:
            self.current_imu = msg
            # Check if robot is stable based on IMU data
            # Simplified stability check
            tilt_threshold = 0.3  # Radians
            tilt_magnitude = math.sqrt(msg.orientation.x**2 + msg.orientation.y**2)
            self.balance_stable = tilt_magnitude < tilt_threshold

    def task_cancel_callback(self, msg):
        """Callback for task cancellation requests"""
        with self.data_lock:
            if self.current_task and self.current_task.id == msg.data:
                self.cancel_current_task()
                self.get_logger().info(f'Cancelled task: {msg.data}')

    def task_execution_callback(self):
        """Main task execution callback"""
        if not self.task_execution_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_execution_time
        self.last_execution_time = current_time

        with self.data_lock:
            # Check for task timeouts
            if self.current_task:
                if current_time > self.current_task.deadline:
                    self.fail_current_task("Task deadline exceeded")
                    self.get_logger().warn(f'Task {self.current_task.id} timed out')

                # Check for step timeouts
                if (self.current_step_start_time > 0 and
                    current_time - self.current_step_start_time > self.step_timeout):
                    self.step_execution_timeout = True

            # Execute current task if available
            if self.current_task and self.current_task.status == ExecutionState.EXECUTING:
                self.execute_current_task_step()

            # Start next task if current is completed
            if (self.current_task and
                self.current_task.status in [ExecutionState.COMPLETED, ExecutionState.FAILED]):
                self.current_task = None
                self.execute_next_task()

        # Publish execution status
        status_msg = String()
        if self.current_task:
            status_msg.data = f"TASK_EXEC: {self.current_task.id}, Step={self.current_task.current_step_index}/{len(self.current_task.steps)}, Status={self.current_task.status.value}"
        else:
            status_msg.data = f"TASK_EXEC: Idle, Queue={len(self.task_queue)}, Completed={len(self.completed_tasks)}"
        self.execution_status_publisher.publish(status_msg)

    def start_task_execution(self, task: Task):
        """Start executing a task"""
        self.current_task = task
        self.current_task.status = ExecutionState.EXECUTING
        self.current_task.current_step_index = 0
        self.current_task.progress = 0.0
        self.current_step_start_time = time.time()
        self.step_execution_timeout = False

        self.get_logger().info(f'Starting execution of task: {task.id}')

    def execute_current_task_step(self):
        """Execute the current step of the current task"""
        if not self.current_task or self.current_task.current_step_index >= len(self.current_task.steps):
            self.current_task.status = ExecutionState.COMPLETED
            self.completed_tasks.append(self.current_task)
            return

        current_step = self.current_task.steps[self.current_task.current_step_index]

        # Check if step timed out
        if self.step_execution_timeout:
            self.get_logger().warn(f'Step {current_step.id} timed out')
            self.fail_current_task("Step execution timed out")
            return

        # Execute the step based on its action
        success = False
        if current_step.action == 'move_to':
            success = self.execute_navigation_step(current_step)
        elif current_step.action == 'grasp':
            success = self.execute_manipulation_step(current_step)
        elif current_step.action == 'speak':
            success = self.execute_speaking_step(current_step)
        elif current_step.action == 'wait':
            success = self.execute_wait_step(current_step)
        elif current_step.action == 'point':
            success = self.execute_pointing_step(current_step)
        else:
            self.get_logger().warn(f'Unknown action: {current_step.action}')
            success = True  # Consider unknown actions as successful to continue

        if success:
            # Move to next step
            self.current_task.current_step_index += 1
            self.current_task.progress = self.current_task.current_step_index / len(self.current_task.steps)
            self.current_step_start_time = time.time()
            self.step_execution_timeout = False

            # Check if task is completed
            if self.current_task.current_step_index >= len(self.current_task.steps):
                self.current_task.status = ExecutionState.COMPLETED
                self.completed_tasks.append(self.current_task)
                self.get_logger().info(f'Task {self.current_task.id} completed successfully')

    def execute_navigation_step(self, step: TaskStep) -> bool:
        """Execute a navigation step"""
        # Extract target location from parameters
        target_x = step.parameters.get('target_x', self.current_pose.position.x)
        target_y = step.parameters.get('target_y', self.current_pose.position.y)
        target_z = step.parameters.get('target_z', self.current_pose.position.z)

        # Calculate direction to target
        dx = target_x - self.current_pose.position.x
        dy = target_y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if we're close enough to target
        arrival_threshold = 0.2  # meters
        if distance < arrival_threshold:
            # Arrived at target
            cmd = Twist()
            self.motion_command_publisher.publish(cmd)  # Stop
            return True

        # Move toward target
        cmd = Twist()
        if distance > 0:
            cmd.linear.x = dx / distance * self.navigation_speed
            cmd.linear.y = dy / distance * self.navigation_speed

        # Add rotation if needed (simplified)
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

        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = f"NAVIGATING: Distance to target: {distance:.2f}m"
        self.feedback_publisher.publish(feedback_msg)

        return False  # Return False to continue execution

    def execute_manipulation_step(self, step: TaskStep) -> bool:
        """Execute a manipulation step"""
        # This would involve complex manipulation planning in a real system
        # For now, we'll simulate simple gripper control

        # Get target object information
        target_object = step.parameters.get('target_object', 'object')

        # Publish joint commands to grasp the object
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'

        # Simulate gripper closing
        joint_cmd.name = ['gripper_joint']  # Placeholder for actual gripper joint
        joint_cmd.position = [0.0]  # Close gripper
        joint_cmd.velocity = [0.1]
        joint_cmd.effort = [10.0]

        self.joint_command_publisher.publish(joint_cmd)

        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = f"MANIPULATING: Attempting to grasp {target_object}"
        self.feedback_publisher.publish(feedback_msg)

        # For simulation, return True after a short time
        if time.time() - self.current_step_start_time > 2.0:
            return True
        else:
            return False

    def execute_speaking_step(self, step: TaskStep) -> bool:
        """Execute a speaking step"""
        text = step.parameters.get('text', 'Hello')

        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = f"SPEAKING: {text}"
        self.feedback_publisher.publish(feedback_msg)

        # In a real system, this would trigger text-to-speech
        # For now, we'll just log it
        self.get_logger().info(f'Speaking: {text}')

        # Return True immediately for speaking
        return True

    def execute_wait_step(self, step: TaskStep) -> bool:
        """Execute a wait step"""
        duration = step.parameters.get('duration', 1.0)

        # Publish feedback
        feedback_msg = String()
        elapsed = time.time() - self.current_step_start_time
        feedback_msg.data = f"WAITING: {elapsed:.1f}/{duration:.1f}s"
        self.feedback_publisher.publish(feedback_msg)

        # Return True when duration has elapsed
        if time.time() - self.current_step_start_time >= duration:
            return True
        else:
            return False

    def execute_pointing_step(self, step: TaskStep) -> bool:
        """Execute a pointing step"""
        target_object = step.parameters.get('target_object', 'object')

        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = f"POINTING: Pointing at {target_object}"
        self.feedback_publisher.publish(feedback_msg)

        # This would involve complex arm kinematics in a real system
        # For simulation, return True after a short time
        if time.time() - self.current_step_start_time > 1.5:
            return True
        else:
            return False

    def get_current_yaw(self) -> float:
        """Get the current yaw angle from the robot's orientation"""
        # Simplified: extract yaw from quaternion
        # In a real system, this would use proper quaternion to euler conversion
        q = self.current_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def execute_next_task(self):
        """Execute the next task in the queue"""
        if self.task_queue:
            next_task = self.task_queue.pop(0)
            self.start_task_execution(next_task)
            self.get_logger().info(f'Started next task: {next_task.id}')
        else:
            # No more tasks in queue
            pass

    def cancel_current_task(self):
        """Cancel the current task"""
        if self.current_task:
            # Stop any ongoing motion
            stop_cmd = Twist()
            self.motion_command_publisher.publish(stop_cmd)

            # Update task status
            self.current_task.status = ExecutionState.FAILED
            self.failed_tasks.append(self.current_task)

            self.get_logger().info(f'Cancelled task: {self.current_task.id}')
            self.current_task = None

    def fail_current_task(self, reason: str = "Unknown error"):
        """Mark the current task as failed"""
        if self.current_task:
            self.current_task.status = ExecutionState.FAILED
            self.failed_tasks.append(self.current_task)
            self.get_logger().error(f'Task {self.current_task.id} failed: {reason}')
            self.current_task = None

    def pause_current_task(self):
        """Pause the current task"""
        if self.current_task:
            self.current_task.status = ExecutionState.PAUSED
            # Stop any ongoing motion
            stop_cmd = Twist()
            self.motion_command_publisher.publish(stop_cmd)
            self.get_logger().info(f'Paused task: {self.current_task.id}')

    def resume_current_task(self):
        """Resume the paused task"""
        if self.current_task and self.current_task.status == ExecutionState.PAUSED:
            self.current_task.status = ExecutionState.EXECUTING
            self.current_step_start_time = time.time()
            self.get_logger().info(f'Resumed task: {self.current_task.id}')

    def enable_task_execution(self, enable: bool):
        """Enable or disable task execution"""
        self.task_execution_enabled = enable
        self.get_logger().info(f"Task execution {'enabled' if enable else 'disabled'}")

    def get_task_stats(self) -> Dict[str, any]:
        """Get task execution statistics"""
        return {
            'current_task': self.current_task.id if self.current_task else None,
            'current_task_status': self.current_task.status.value if self.current_task else 'none',
            'queue_size': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'enabled': self.task_execution_enabled
        }

    def get_current_task_info(self) -> Optional[Dict[str, any]]:
        """Get information about the current task"""
        if not self.current_task:
            return None

        return {
            'id': self.current_task.id,
            'type': self.current_task.type.value,
            'status': self.current_task.status.value,
            'progress': self.current_task.progress,
            'current_step': self.current_task.current_step_index,
            'total_steps': len(self.current_task.steps),
            'created_time': self.current_task.created_time
        }

    def emergency_stop(self):
        """Emergency stop - stop all motion and cancel current task"""
        with self.data_lock:
            # Stop any ongoing motion
            stop_cmd = Twist()
            self.motion_command_publisher.publish(stop_cmd)

            # Cancel current task if any
            if self.current_task:
                self.current_task.status = ExecutionState.FAILED
                self.failed_tasks.append(self.current_task)
                self.current_task = None

            # Clear task queue
            self.task_queue.clear()

            self.get_logger().error('Emergency stop activated - all tasks cancelled')


def main(args=None):
    rclpy.init(args=args)

    task_execution_node = TaskExecutionNode()

    try:
        rclpy.spin(task_execution_node)
    except KeyboardInterrupt:
        # Emergency stop on shutdown
        task_execution_node.emergency_stop()
        pass
    finally:
        task_execution_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()