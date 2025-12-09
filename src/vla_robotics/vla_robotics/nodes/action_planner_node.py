#!/usr/bin/env python3

"""
Action Planner Node for Humanoid Robot VLA System

This node plans actions based on vision-language input, creating
executable action sequences from natural language commands and
visual scene understanding.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from geometry_msgs.msg import Twist, Pose, Point, Vector3
from sensor_msgs.msg import JointState
from nav_msgs.msg import Path
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import re


class ActionType(Enum):
    MOVE_TO = "move_to"
    GRASP = "grasp"
    RELEASE = "release"
    POINT = "point"
    SPEAK = "speak"
    WAIT = "wait"
    APPROACH = "approach"
    AVOID = "avoid"


@dataclass
class ActionStep:
    """Represents a single action step in an action sequence"""
    id: str
    type: ActionType
    parameters: Dict[str, any]
    priority: int
    estimated_time: float
    dependencies: List[str]


@dataclass
class ActionPlan:
    """Represents a complete action plan"""
    id: str
    steps: List[ActionStep]
    created_time: float
    estimated_duration: float
    status: str  # 'planned', 'in_progress', 'completed', 'failed'


class ActionPlannerNode(Node):
    def __init__(self):
        super().__init__('action_planner_node')

        # Publishers for action planning
        self.action_plan_publisher = self.create_publisher(String, 'vla/action_plan', 10)
        self.motion_command_publisher = self.create_publisher(Twist, 'motion_commands', 10)
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.action_status_publisher = self.create_publisher(String, 'vla/action_status', 10)
        self.path_publisher = self.create_publisher(Path, 'vla/planned_path', 10)

        # Subscribers for vision-language input
        self.vision_language_command_subscriber = self.create_subscription(
            String, 'vla/language_command', self.vision_language_command_callback, 10)
        self.scene_description_subscriber = self.create_subscription(
            String, 'vla/scene_description', self.scene_description_callback, 10)
        self.object_grounding_subscriber = self.create_subscription(
            String, 'vla/object_grounding', self.object_grounding_callback, 10)
        self.current_pose_subscriber = self.create_subscription(
            Pose, 'current_pose', self.current_pose_callback, 10)

        # Timer for action planning
        self.action_timer = self.create_timer(0.5, self.action_planning_callback)  # 2 Hz

        # Action planning state
        self.current_command = ""
        self.current_scene_description = ""
        self.current_object_info = {}
        self.current_pose = Pose()
        self.action_plans = []
        self.current_plan: Optional[ActionPlan] = None
        self.current_step_index = 0
        self.action_planning_enabled = True
        self.sim_time = time.time()
        self.last_action_time = time.time()

        # Action planning parameters
        self.max_plan_steps = 10
        self.default_action_time = 2.0  # seconds per action
        self.safety_margin = 0.5  # meters for navigation safety

        # Language command patterns
        self.command_patterns = {
            'move': [r'go to', r'move to', r'walk to', r'navigate to', r'approach'],
            'grasp': [r'pick up', r'grasp', r'grab', r'take', r'lift'],
            'release': [r'put down', r'release', r'drop', r'place'],
            'point': [r'point to', r'point at', r'indicate', r'show'],
            'speak': [r'say', r'tell', r'speak'],
            'wait': [r'wait', r'pause', r'stop']
        }

        # Object reference resolution
        self.object_resolutions = {}  # Maps language references to actual objects

        # Threading lock for data access
        self.data_lock = threading.Lock()

        self.get_logger().info('Action Planner Node initialized')

    def vision_language_command_callback(self, msg):
        """Callback for vision-language commands"""
        with self.data_lock:
            self.current_command = msg.data
            self.get_logger().info(f'Received action command: {msg.data}')

            # Generate a new action plan based on the command
            if self.current_command:
                self.generate_action_plan()

    def scene_description_callback(self, msg):
        """Callback for scene descriptions"""
        with self.data_lock:
            self.current_scene_description = msg.data

    def object_grounding_callback(self, msg):
        """Callback for object grounding information"""
        # This would normally process the object grounding message
        # For now, we'll just log it
        pass

    def current_pose_callback(self, msg):
        """Callback for current robot pose"""
        with self.data_lock:
            self.current_pose = msg

    def action_planning_callback(self):
        """Main action planning callback"""
        if not self.action_planning_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_action_time
        self.last_action_time = current_time

        with self.data_lock:
            # Execute current action plan if available
            if self.current_plan and self.current_step_index < len(self.current_plan.steps):
                self.execute_current_action_step()

            # Check if current plan is completed
            if (self.current_plan and
                self.current_step_index >= len(self.current_plan.steps)):
                self.current_plan.status = 'completed'
                self.current_plan = None
                self.current_step_index = 0

        # Publish action status
        status_msg = String()
        plan_status = self.current_plan.status if self.current_plan else 'idle'
        step_info = f"Step {self.current_step_index}/{len(self.current_plan.steps) if self.current_plan else 0}" if self.current_plan else "No plan"
        status_msg.data = f"ACTION_PLAN: {plan_status}, {step_info}, Command='{self.current_command[:30]}...'"
        self.action_status_publisher.publish(status_msg)

    def generate_action_plan(self):
        """Generate an action plan based on the current command"""
        command = self.current_command.lower()

        # Parse the command to identify the main action and target
        action_steps = []

        # Identify action type from command
        action_type = self.identify_action_type(command)

        if action_type == ActionType.MOVE_TO:
            # Extract target location from command
            target_location = self.extract_location_from_command(command)
            if target_location:
                # Create navigation action
                nav_step = ActionStep(
                    id=f"move_to_{len(action_steps)}",
                    type=ActionType.MOVE_TO,
                    parameters={'target_location': target_location},
                    priority=1,
                    estimated_time=self.default_action_time,
                    dependencies=[]
                )
                action_steps.append(nav_step)

        elif action_type == ActionType.GRASP:
            # Extract target object from command
            target_object = self.extract_object_from_command(command)
            if target_object:
                # Create approach action
                approach_step = ActionStep(
                    id=f"approach_{len(action_steps)}",
                    type=ActionType.APPROACH,
                    parameters={'target_object': target_object},
                    priority=1,
                    estimated_time=self.default_action_time,
                    dependencies=[]
                )
                action_steps.append(approach_step)

                # Create grasp action
                grasp_step = ActionStep(
                    id=f"grasp_{len(action_steps)}",
                    type=ActionType.GRASP,
                    parameters={'target_object': target_object},
                    priority=2,
                    estimated_time=self.default_action_time,
                    dependencies=[approach_step.id]
                )
                action_steps.append(grasp_step)

        elif action_type == ActionType.RELEASE:
            # Extract placement location from command
            placement_location = self.extract_location_from_command(command)
            if placement_location:
                # Create placement action
                release_step = ActionStep(
                    id=f"release_{len(action_steps)}",
                    type=ActionType.RELEASE,
                    parameters={'placement_location': placement_location},
                    priority=1,
                    estimated_time=self.default_action_time,
                    dependencies=[]
                )
                action_steps.append(release_step)

        elif action_type == ActionType.POINT:
            # Extract target object from command
            target_object = self.extract_object_from_command(command)
            if target_object:
                # Create pointing action
                point_step = ActionStep(
                    id=f"point_{len(action_steps)}",
                    type=ActionType.POINT,
                    parameters={'target_object': target_object},
                    priority=1,
                    estimated_time=self.default_action_time,
                    dependencies=[]
                )
                action_steps.append(point_step)

        # Create the action plan
        if action_steps:
            plan = ActionPlan(
                id=f"plan_{len(self.action_plans)}",
                steps=action_steps,
                created_time=time.time(),
                estimated_duration=len(action_steps) * self.default_action_time,
                status='planned'
            )

            self.action_plans.append(plan)
            self.current_plan = plan
            self.current_step_index = 0

            # Publish the plan
            plan_msg = String()
            plan_msg.data = f"PLAN: {plan.id}, Steps: {[step.type.value for step in plan.steps]}"
            self.action_plan_publisher.publish(plan_msg)

            self.get_logger().info(f'Generated action plan with {len(action_steps)} steps')

    def identify_action_type(self, command: str) -> ActionType:
        """Identify the primary action type from a command"""
        for action_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command):
                    return ActionType(action_type)

        # Default to move if no specific action is identified
        return ActionType.MOVE_TO

    def extract_location_from_command(self, command: str) -> Optional[Dict[str, float]]:
        """Extract location information from command"""
        # Simple location extraction
        # In a real system, this would use more sophisticated NLP
        location_keywords = {
            'kitchen': {'x': 2.0, 'y': 1.0, 'z': 0.0},
            'living room': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'bedroom': {'x': -2.0, 'y': 1.0, 'z': 0.0},
            'table': {'x': 1.0, 'y': 0.0, 'z': 0.0},
            'chair': {'x': 0.5, 'y': 0.5, 'z': 0.0}
        }

        for keyword, location in location_keywords.items():
            if keyword in command:
                return location

        # Try to extract numeric coordinates
        # Look for patterns like "go to x=1.0, y=2.0"
        coord_match = re.search(r'x[=\s]*([+-]?\d*\.?\d+)[,\s]*y[=\s]*([+-]?\d*\.?\d+)', command)
        if coord_match:
            x = float(coord_match.group(1))
            y = float(coord_match.group(2))
            return {'x': x, 'y': y, 'z': 0.0}

        return None

    def extract_object_from_command(self, command: str) -> Optional[str]:
        """Extract object information from command"""
        # Simple object extraction
        # In a real system, this would use object detection results
        object_keywords = [
            'red ball', 'blue cup', 'green box', 'yellow object',
            'object', 'thing', 'item', 'cup', 'ball', 'box', 'bottle'
        ]

        for keyword in object_keywords:
            if keyword in command:
                return keyword

        return None

    def execute_current_action_step(self):
        """Execute the current action step"""
        if not self.current_plan or self.current_step_index >= len(self.current_plan.steps):
            return

        current_step = self.current_plan.steps[self.current_step_index]

        if current_step.type == ActionType.MOVE_TO:
            self.execute_move_to_action(current_step)
        elif current_step.type == ActionType.GRASP:
            self.execute_grasp_action(current_step)
        elif current_step.type == ActionType.RELEASE:
            self.execute_release_action(current_step)
        elif current_step.type == ActionType.POINT:
            self.execute_point_action(current_step)
        elif current_step.type == ActionType.SPEAK:
            self.execute_speak_action(current_step)
        elif current_step.type == ActionType.WAIT:
            self.execute_wait_action(current_step)
        elif current_step.type == ActionType.APPROACH:
            self.execute_approach_action(current_step)
        elif current_step.type == ActionType.AVOID:
            self.execute_avoid_action(current_step)

        # Move to next step after execution
        self.current_step_index += 1

    def execute_move_to_action(self, step: ActionStep):
        """Execute a move-to action"""
        target_location = step.parameters.get('target_location')
        if target_location:
            # Create motion command to move to target
            cmd = Twist()

            # Calculate direction to target
            dx = target_location['x'] - self.current_pose.position.x
            dy = target_location['y'] - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > 0.1:  # If not already at target
                cmd.linear.x = dx / distance * 0.3  # Move at 0.3 m/s
                cmd.linear.y = dy / distance * 0.3
            else:
                cmd.linear.x = 0.0
                cmd.linear.y = 0.0

            self.motion_command_publisher.publish(cmd)
            self.get_logger().info(f'Moving to location: ({target_location["x"]}, {target_location["y"]})')

    def execute_grasp_action(self, step: ActionStep):
        """Execute a grasp action"""
        target_object = step.parameters.get('target_object')
        if target_object:
            # Create joint commands for grasping
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.get_clock().now().to_msg()
            joint_cmd.header.frame_id = 'base_link'

            # Add joint names and positions for grasping
            # This is simplified - in reality would depend on robot kinematics
            joint_cmd.name = ['gripper_joint']  # Placeholder for actual gripper joint
            joint_cmd.position = [0.5]  # Close gripper
            joint_cmd.velocity = [0.0]
            joint_cmd.effort = [0.0]

            self.joint_command_publisher.publish(joint_cmd)
            self.get_logger().info(f'Grasping object: {target_object}')

    def execute_release_action(self, step: ActionStep):
        """Execute a release action"""
        placement_location = step.parameters.get('placement_location')
        if placement_location:
            # Create joint commands for releasing
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.get_clock().now().to_msg()
            joint_cmd.header.frame_id = 'base_link'

            # Open gripper
            joint_cmd.name = ['gripper_joint']  # Placeholder for actual gripper joint
            joint_cmd.position = [0.0]  # Open gripper
            joint_cmd.velocity = [0.0]
            joint_cmd.effort = [0.0]

            self.joint_command_publisher.publish(joint_cmd)
            self.get_logger().info(f'Releasing object at: ({placement_location["x"]}, {placement_location["y"]})')

    def execute_point_action(self, step: ActionStep):
        """Execute a point action"""
        target_object = step.parameters.get('target_object')
        if target_object:
            # Create joint commands to point at the object
            # This is simplified - in reality would involve complex kinematics
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.get_clock().now().to_msg()
            joint_cmd.header.frame_id = 'base_link'

            # Placeholder joint commands for pointing
            joint_cmd.name = ['arm_joint_1', 'arm_joint_2']  # Placeholder joints
            joint_cmd.position = [0.5, 0.3]  # Placeholder positions
            joint_cmd.velocity = [0.0, 0.0]
            joint_cmd.effort = [0.0, 0.0]

            self.joint_command_publisher.publish(joint_cmd)
            self.get_logger().info(f'Pointing at object: {target_object}')

    def execute_speak_action(self, step: ActionStep):
        """Execute a speak action"""
        text = step.parameters.get('text', 'Hello')
        self.get_logger().info(f'Speaking: {text}')

    def execute_wait_action(self, step: ActionStep):
        """Execute a wait action"""
        duration = step.parameters.get('duration', 1.0)
        self.get_logger().info(f'Waiting for {duration} seconds')

    def execute_approach_action(self, step: ActionStep):
        """Execute an approach action"""
        target_object = step.parameters.get('target_object')
        if target_object:
            # Move closer to the object
            cmd = Twist()
            cmd.linear.x = 0.2  # Move forward slowly
            cmd.angular.z = 0.0
            self.motion_command_publisher.publish(cmd)
            self.get_logger().info(f'Approaching object: {target_object}')

    def execute_avoid_action(self, step: ActionStep):
        """Execute an avoid action"""
        # In a real system, this would use obstacle detection
        cmd = Twist()
        cmd.linear.x = -0.2  # Move backward
        cmd.angular.z = 0.5  # Turn to avoid
        self.motion_command_publisher.publish(cmd)
        self.get_logger().info('Avoiding obstacle')

    def plan_navigation_to_object(self, object_name: str) -> Optional[ActionPlan]:
        """Plan navigation actions to reach a specific object"""
        # This would integrate with perception system to find object location
        # For now, return a simple navigation plan
        nav_steps = [
            ActionStep(
                id="navigate_to_object",
                type=ActionType.MOVE_TO,
                parameters={'target_object': object_name},
                priority=1,
                estimated_time=self.default_action_time,
                dependencies=[]
            )
        ]

        plan = ActionPlan(
            id=f"nav_plan_{object_name}",
            steps=nav_steps,
            created_time=time.time(),
            estimated_duration=self.default_action_time,
            status='planned'
        )

        return plan

    def plan_manipulation_for_object(self, object_name: str, action: str) -> Optional[ActionPlan]:
        """Plan manipulation actions for a specific object"""
        steps = []

        # Approach object
        approach_step = ActionStep(
            id=f"approach_{object_name}",
            type=ActionType.APPROACH,
            parameters={'target_object': object_name},
            priority=1,
            estimated_time=self.default_action_time,
            dependencies=[]
        )
        steps.append(approach_step)

        # Perform action (grasp or release)
        if action == 'grasp':
            grasp_step = ActionStep(
                id=f"grasp_{object_name}",
                type=ActionType.GRASP,
                parameters={'target_object': object_name},
                priority=2,
                estimated_time=self.default_action_time,
                dependencies=[approach_step.id]
            )
            steps.append(grasp_step)
        elif action == 'release':
            release_step = ActionStep(
                id=f"release_{object_name}",
                type=ActionType.RELEASE,
                parameters={'target_object': object_name},
                priority=2,
                estimated_time=self.default_action_time,
                dependencies=[approach_step.id]
            )
            steps.append(release_step)

        plan = ActionPlan(
            id=f"manip_plan_{object_name}_{action}",
            steps=steps,
            created_time=time.time(),
            estimated_duration=len(steps) * self.default_action_time,
            status='planned'
        )

        return plan

    def cancel_current_plan(self):
        """Cancel the current action plan"""
        with self.data_lock:
            if self.current_plan:
                self.current_plan.status = 'failed'
                self.current_plan = None
                self.current_step_index = 0
                self.get_logger().info('Current action plan cancelled')

    def enable_action_planning(self, enable: bool):
        """Enable or disable action planning"""
        self.action_planning_enabled = enable
        self.get_logger().info(f"Action planning {'enabled' if enable else 'disabled'}")

    def get_action_plan_stats(self) -> Dict[str, any]:
        """Get action planning statistics"""
        return {
            'total_plans': len(self.action_plans),
            'current_plan': self.current_plan.id if self.current_plan else None,
            'current_step': self.current_step_index,
            'command_queue_size': len([p for p in self.action_plans if p.status == 'planned']),
            'enabled': self.action_planning_enabled
        }


def main(args=None):
    rclpy.init(args=args)

    action_planner_node = ActionPlannerNode()

    try:
        rclpy.spin(action_planner_node)
    except KeyboardInterrupt:
        pass
    finally:
        action_planner_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()