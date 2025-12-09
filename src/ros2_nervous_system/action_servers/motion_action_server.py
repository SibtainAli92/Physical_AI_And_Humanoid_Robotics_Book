#!/usr/bin/env python3
# motion_action_server.py
# Action server for long-running motion operations

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import threading
import time
import math
from typing import Optional

# Import action messages (these would normally be in your action definition package)
# For this example, I'll define simplified versions of common action messages
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from nav_msgs.action import NavigateToPose


class FollowJointTrajectoryActionServer(Node):
    """
    Action server for following joint trajectories
    """
    def __init__(self):
        super().__init__('follow_joint_trajectory_action_server')

        # Declare parameters
        self.declare_parameter('execution_timeout', 30.0)  # seconds
        self.declare_parameter('goal_tolerance', 0.01)     # radians
        self.declare_parameter('velocity_tolerance', 0.1)  # rad/s

        self.execution_timeout = self.get_parameter('execution_timeout').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.velocity_tolerance = self.get_parameter('velocity_tolerance').value

        # Create action server
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'follow_joint_trajectory',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10)
        )

        self.joint_command_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            QoSProfile(depth=10)
        )

        # Internal state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self._goal_handle = None
        self._goal_lock = threading.Lock()

        self.get_logger().info('Follow Joint Trajectory Action Server initialized')

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """
        Accept or reject a client request to begin an action
        """
        self.get_logger().info('Received goal request')

        # Check if trajectory is valid
        trajectory = goal_request.trajectory
        if len(trajectory.joint_names) == 0:
            self.get_logger().warn('Goal rejected: empty joint names')
            return GoalResponse.REJECT

        if len(trajectory.points) == 0:
            self.get_logger().warn('Goal rejected: empty trajectory points')
            return GoalResponse.REJECT

        # Check if we can accept the goal
        with self._goal_lock:
            if self._goal_handle is not None and self._goal_handle.is_active:
                # Reject if another goal is already active
                self.get_logger().info('Goal rejected: another goal is active')
                return GoalResponse.REJECT

        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept or reject a client request to cancel an action
        """
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def joint_state_callback(self, msg: JointState):
        """
        Callback for joint state messages
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def execute_callback(self, goal_handle):
        """
        Execute the requested action
        """
        self.get_logger().info('Executing goal...')

        with self._goal_lock:
            self._goal_handle = goal_handle

        # Get the trajectory from the goal
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names

        # Feedback and result messages
        feedback_msg = FollowJointTrajectory.Feedback()
        result_msg = FollowJointTrajectory.Result()

        # Initialize feedback
        feedback_msg.joint_names = joint_names
        feedback_msg.actual.positions = [0.0] * len(joint_names)
        feedback_msg.actual.velocities = [0.0] * len(joint_names)
        feedback_msg.desired.positions = [0.0] * len(joint_names)
        feedback_msg.error.positions = [0.0] * len(joint_names)

        try:
            # Execute the trajectory point by point
            for i, point in enumerate(trajectory.points):
                # Check if the goal was canceled
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    goal_handle.canceled()
                    self.get_logger().info('Goal canceled')
                    return result_msg

                # Update goal state
                goal_handle.publish_feedback(feedback_msg)

                # Send trajectory point to controller
                trajectory_msg = JointTrajectory()
                trajectory_msg.joint_names = joint_names
                trajectory_msg.points = [point]  # Send one point at a time
                self.joint_command_pub.publish(trajectory_msg)

                # Wait for the specified time for this trajectory point
                time.sleep(point.time_from_start.sec + point.time_from_start.nanosec / 1e9)

                # Update feedback with current state
                current_positions = []
                current_velocities = []
                desired_positions = []
                errors = []

                for j, joint_name in enumerate(joint_names):
                    current_pos = self.current_joint_positions.get(joint_name, 0.0)
                    current_vel = self.current_joint_velocities.get(joint_name, 0.0)
                    desired_pos = point.positions[j] if j < len(point.positions) else 0.0

                    current_positions.append(current_pos)
                    current_velocities.append(current_vel)
                    desired_positions.append(desired_pos)
                    errors.append(abs(current_pos - desired_pos))

                feedback_msg.actual.positions = current_positions
                feedback_msg.actual.velocities = current_velocities
                feedback_msg.desired.positions = desired_positions
                feedback_msg.error.positions = errors

                # Check tolerances
                max_error = max(errors) if errors else 0.0
                if max_error > self.goal_tolerance:
                    self.get_logger().warn(f'Trajectory tolerance exceeded: {max_error} > {self.goal_tolerance}')
                    # Could implement tolerance checking here based on the action specification

            # Check final position
            final_positions = trajectory.points[-1].positions
            final_errors = []
            for i, joint_name in enumerate(joint_names):
                if i < len(final_positions):
                    current_pos = self.current_joint_positions.get(joint_name, 0.0)
                    error = abs(current_pos - final_positions[i])
                    final_errors.append(error)

            max_final_error = max(final_errors) if final_errors else 0.0

            if max_final_error <= self.goal_tolerance:
                goal_handle.succeed()
                result_msg.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                self.get_logger().info('Goal succeeded')
            else:
                goal_handle.abort()
                result_msg.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                self.get_logger().info('Goal aborted due to tolerance violation')

        except Exception as e:
            self.get_logger().error(f'Error executing goal: {str(e)}')
            goal_handle.abort()
            result_msg.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result_msg.error_string = str(e)

        finally:
            with self._goal_lock:
                self._goal_handle = None

        return result_msg


class NavigateToPoseActionServer(Node):
    """
    Action server for navigation to a pose
    """
    def __init__(self):
        super().__init__('navigate_to_pose_action_server')

        # Declare parameters
        self.declare_parameter('linear_tolerance', 0.1)    # meters
        self.declare_parameter('angular_tolerance', 0.1)   # radians
        self.declare_parameter('max_linear_speed', 0.5)    # m/s
        self.declare_parameter('max_angular_speed', 0.5)   # rad/s

        self.linear_tolerance = self.get_parameter('linear_tolerance').value
        self.angular_tolerance = self.get_parameter('angular_tolerance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value

        # Create action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers and subscribers for navigation
        self.current_pose_sub = self.create_subscription(
            Pose,
            '/current_pose',
            self.current_pose_callback,
            QoSProfile(depth=10)
        )

        self.cmd_vel_pub = self.create_publisher(
            Point,  # Using Point as a simplified velocity command
            '/cmd_vel',
            QoSProfile(depth=10)
        )

        # Internal state
        self.current_pose = Pose()
        self._goal_handle = None
        self._goal_lock = threading.Lock()

        self.get_logger().info('Navigate To Pose Action Server initialized')

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """
        Accept or reject a client request to begin an action
        """
        self.get_logger().info('Received navigation goal request')

        # Check if goal pose is valid
        goal_pose = goal_request.pose
        if goal_pose.position.x == 0.0 and goal_pose.position.y == 0.0 and goal_pose.position.z == 0.0:
            self.get_logger().warn('Goal rejected: invalid goal pose')
            return GoalResponse.REJECT

        self.get_logger().info('Navigation goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept or reject a client request to cancel an action
        """
        self.get_logger().info('Received navigation cancel request')
        return CancelResponse.ACCEPT

    def current_pose_callback(self, msg: Pose):
        """
        Callback for current pose messages
        """
        self.current_pose = msg

    def execute_callback(self, goal_handle):
        """
        Execute the requested navigation action
        """
        self.get_logger().info('Executing navigation goal...')

        with self._goal_lock:
            self._goal_handle = goal_handle

        # Get the goal pose
        goal_pose = goal_handle.request.pose

        # Feedback and result messages
        feedback_msg = NavigateToPose.Feedback()
        result_msg = NavigateToPose.Result()

        # Initialize feedback
        feedback_msg.current_pose = self.current_pose

        try:
            # Navigation loop
            while rclpy.ok():
                # Check if the goal was canceled
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().info('Navigation goal canceled')
                    result_msg.result = 0  # Canceled result
                    return result_msg

                # Calculate distance to goal
                dx = goal_pose.position.x - self.current_pose.position.x
                dy = goal_pose.position.y - self.current_pose.position.y
                distance = math.sqrt(dx*dx + dy*dy)

                # Calculate angular error
                # This is a simplified approach - in reality you'd need proper orientation calculation
                angle_error = 0.0  # Simplified for this example

                # Check if we've reached the goal
                if distance < self.linear_tolerance and abs(angle_error) < self.angular_tolerance:
                    goal_handle.succeed()
                    self.get_logger().info('Navigation goal succeeded')
                    result_msg.result = 1  # Success result
                    return result_msg

                # Calculate velocity commands (simplified)
                linear_vel = min(self.max_linear_speed, distance * 0.5)  # Simple proportional control
                angular_vel = min(self.max_angular_speed, abs(angle_error) * 0.5)

                # Create and publish velocity command
                vel_cmd = Point()
                vel_cmd.x = linear_vel
                vel_cmd.y = angular_vel
                vel_cmd.z = 0.0
                self.cmd_vel_pub.publish(vel_cmd)

                # Update feedback
                feedback_msg.current_pose = self.current_pose
                feedback_msg.distance_remaining = distance
                goal_handle.publish_feedback(feedback_msg)

                # Sleep briefly to allow other callbacks to run
                time.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f'Error executing navigation: {str(e)}')
            goal_handle.abort()
            result_msg.result = -1  # Error result
            result_msg.message = str(e)

        finally:
            # Stop the robot when done
            stop_cmd = Point()
            stop_cmd.x = 0.0
            stop_cmd.y = 0.0
            stop_cmd.z = 0.0
            self.cmd_vel_pub.publish(stop_cmd)

            with self._goal_lock:
                self._goal_handle = None

        return result_msg


def main(args=None):
    rclpy.init(args=args)

    # Create action servers
    joint_traj_server = FollowJointTrajectoryActionServer()
    nav_server = NavigateToPoseActionServer()

    # Use a MultiThreadedExecutor to handle callbacks from multiple servers
    executor = MultiThreadedExecutor()
    executor.add_node(joint_traj_server)
    executor.add_node(nav_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        joint_traj_server.get_logger().info('Joint trajectory server interrupted by user')
        nav_server.get_logger().info('Navigation server interrupted by user')
    finally:
        joint_traj_server.destroy()
        nav_server.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()