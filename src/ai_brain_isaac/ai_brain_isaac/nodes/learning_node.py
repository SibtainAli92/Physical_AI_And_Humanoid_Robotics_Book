#!/usr/bin/env python3

"""
Learning Node for Humanoid Robot AI Brain

This node handles machine learning, adaptation, and continuous improvement
using techniques similar to those in NVIDIA Isaac.
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
import pickle
import os


@dataclass
class Experience:
    """Represents a learning experience tuple"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float


class LearningNode(Node):
    def __init__(self):
        super().__init__('learning_node')

        # Publishers for learning data
        self.learning_status_publisher = self.create_publisher(String, 'ai/learning_status', 10)
        self.performance_publisher = self.create_publisher(Float32, 'ai/performance', 10)

        # Subscribers for robot state and feedback
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.command_subscriber = self.create_subscription(
            Twist, 'motion_commands', self.command_callback, 10)
        self.reward_subscriber = self.create_subscription(
            Float32, 'ai/reward', self.reward_callback, 10)

        # Timer for learning updates
        self.learning_timer = self.create_timer(0.5, self.learning_callback)  # 2 Hz

        # Learning state
        self.current_state = np.zeros(12)  # Simplified state vector
        self.previous_state = np.zeros(12)
        self.current_action = np.zeros(3)  # Simplified action vector (linear x, y, angular z)
        self.current_reward = 0.0
        self.episode_reward = 0.0
        self.total_reward = 0.0
        self.experience_buffer = []
        self.max_experience_buffer_size = 10000

        # Learning parameters
        self.learning_enabled = True
        self.training_mode = True
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.exploration_rate = 0.1
        self.update_frequency = 10  # Update every 10 experiences
        self.episode_count = 0
        self.step_count = 0

        # Simple Q-table for learning (simplified approach)
        self.q_table = {}
        self.state_action_space_size = 1000  # Discretized state-action space

        # Performance tracking
        self.performance_history = []
        self.average_performance = 0.0
        self.last_update_time = time.time()

        # Threading lock for data access
        self.data_lock = threading.Lock()

        # Load any saved model
        self.load_model()

        self.get_logger().info('Learning Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        with self.data_lock:
            # Extract relevant joint information for state representation
            joint_positions = []
            joint_velocities = []

            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    joint_positions.append(msg.position[i])
                if i < len(msg.velocity):
                    joint_velocities.append(msg.velocity[i])

            # Create simplified state representation
            # Use first 6 joint positions and velocities (if available)
            state_features = []
            for pos in joint_positions[:6]:
                state_features.append(pos)
            for vel in joint_velocities[:6]:
                state_features.append(vel)

            # Pad or truncate to fixed size
            while len(state_features) < 12:
                state_features.append(0.0)
            state_features = state_features[:12]

            self.previous_state = self.current_state.copy()
            self.current_state = np.array(state_features)

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.data_lock:
            # Add position and velocity to state
            self.current_state[6] = msg.pose.pose.position.x
            self.current_state[7] = msg.pose.pose.position.y
            self.current_state[8] = msg.twist.twist.linear.x
            self.current_state[9] = msg.twist.twist.linear.y
            self.current_state[10] = msg.twist.twist.angular.z

    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.data_lock:
            # Add IMU data to state
            self.current_state[11] = msg.linear_acceleration.z  # Simplified

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        with self.data_lock:
            # Process laser scan for state representation
            # For simplicity, just use a few key ranges
            if len(msg.ranges) >= 3:
                self.current_state[4] = msg.ranges[0]  # Front
                self.current_state[5] = msg.ranges[len(msg.ranges)//2]  # Center

    def command_callback(self, msg):
        """Callback for motion commands (actions)"""
        with self.data_lock:
            # Store the action taken
            self.current_action = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    def reward_callback(self, msg):
        """Callback for reward signals"""
        with self.data_lock:
            self.current_reward = msg.data
            self.episode_reward += msg.data
            self.total_reward += msg.data

    def learning_callback(self):
        """Main learning callback"""
        if not self.learning_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        with self.data_lock:
            # Store experience
            experience = Experience(
                state=self.previous_state.copy(),
                action=self.current_action.copy(),
                reward=self.current_reward,
                next_state=self.current_state.copy(),
                done=False,  # Simplified - in real system would need to determine episode boundaries
                timestamp=current_time
            )

            self.experience_buffer.append(experience)
            if len(self.experience_buffer) > self.max_experience_buffer_size:
                self.experience_buffer.pop(0)  # Remove oldest experience

            # Update step count
            self.step_count += 1

            # Perform learning update periodically
            if self.step_count % self.update_frequency == 0 and len(self.experience_buffer) > 10:
                if self.training_mode:
                    self.update_model()

            # Calculate performance metrics
            self.update_performance_metrics()

        # Publish learning status
        status_msg = String()
        status_msg.data = f"LEARNING: Buffer={len(self.experience_buffer)}, Reward={self.episode_reward:.2f}, Performance={self.average_performance:.3f}"
        self.learning_status_publisher.publish(status_msg)

        # Publish performance
        perf_msg = Float32()
        perf_msg.data = self.average_performance
        self.performance_publisher.publish(perf_msg)

    def update_model(self):
        """Update the learning model using experiences in buffer"""
        if not self.experience_buffer:
            return

        # Simplified Q-learning update
        # In a real system, this would use neural networks and more sophisticated algorithms
        for experience in self.experience_buffer[-10:]:  # Use last 10 experiences
            # Discretize state and action for Q-table approach
            state_key = self.discretize_state(experience.state)
            action_key = self.discretize_action(experience.action)

            # Get current Q-value
            q_value = self.q_table.get((state_key, action_key), 0.0)

            # Calculate target Q-value
            next_state_key = self.discretize_state(experience.next_state)
            next_max_q = max([self.q_table.get((next_state_key, a), 0.0) for a in range(3)], default=0.0)
            target_q = experience.reward + self.discount_factor * next_max_q

            # Update Q-value
            td_error = target_q - q_value
            new_q = q_value + self.learning_rate * td_error
            self.q_table[(state_key, action_key)] = new_q

    def discretize_state(self, state: np.ndarray) -> int:
        """Discretize continuous state for Q-table approach"""
        # Simple discretization - in reality would use more sophisticated methods
        state_discrete = (state * 10).astype(int)  # Scale and discretize
        # Create a hash-like value from state components
        hash_val = 0
        for i, val in enumerate(state_discrete):
            hash_val += val * (31 ** i)  # Polynomial rolling hash
        return abs(hash_val) % self.state_action_space_size

    def discretize_action(self, action: np.ndarray) -> int:
        """Discretize continuous action for Q-table approach"""
        # Map continuous action to discrete action space
        # For simplicity, map to 3 actions: forward, turn left, turn right
        if abs(action[0]) > abs(action[2]):  # Linear motion dominates
            if action[0] > 0.1:
                return 0  # Move forward
            elif action[0] < -0.1:
                return 1  # Move backward
            else:
                return 2  # No linear motion
        else:  # Angular motion dominates
            if action[2] > 0.1:
                return 3  # Turn left
            elif action[2] < -0.1:
                return 4  # Turn right
            else:
                return 2  # No angular motion

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action based on current state and learned policy"""
        if not self.learning_enabled:
            # Return a default action if learning is disabled
            return np.array([0.0, 0.0, 0.0])

        state_key = self.discretize_state(state)

        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Random action (exploration)
            return np.random.uniform(-1, 1, size=3)
        else:
            # Greedy action based on Q-table (exploitation)
            best_action = 0
            best_q = float('-inf')
            for action_idx in range(5):  # 5 discrete actions
                q_val = self.q_table.get((state_key, action_idx), 0.0)
                if q_val > best_q:
                    best_q = q_val
                    best_action = action_idx

            # Map discrete action back to continuous
            action = np.array([0.0, 0.0, 0.0])
            if best_action == 0:  # Move forward
                action[0] = 0.5
            elif best_action == 1:  # Move backward
                action[0] = -0.5
            elif best_action == 2:  # No motion
                pass  # Already zero
            elif best_action == 3:  # Turn left
                action[2] = 0.5
            elif best_action == 4:  # Turn right
                action[2] = -0.5

            return action

    def update_performance_metrics(self):
        """Update performance metrics based on recent experiences"""
        # Calculate performance as average reward over recent episodes
        recent_rewards = [exp.reward for exp in self.experience_buffer[-100:]]
        if recent_rewards:
            self.average_performance = sum(recent_rewards) / len(recent_rewards)
            self.performance_history.append(self.average_performance)
            if len(self.performance_history) > 50:  # Keep last 50 performance values
                self.performance_history.pop(0)

    def enable_learning(self, enable: bool):
        """Enable or disable learning"""
        self.learning_enabled = enable
        self.get_logger().info(f"Learning {'enabled' if enable else 'disabled'}")

    def set_training_mode(self, training: bool):
        """Set training or inference mode"""
        self.training_mode = training
        self.get_logger().info(f"Learning mode set to {'training' if training else 'inference'}")

    def reset_episode(self):
        """Reset for a new learning episode"""
        with self.data_lock:
            self.episode_reward = 0.0
            self.episode_count += 1

    def save_model(self, filepath: str = "learning_model.pkl"):
        """Save the learning model to file"""
        model_data = {
            'q_table': self.q_table,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'average_performance': self.average_performance
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            self.get_logger().info(f"Model saved to {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error saving model: {e}")

    def load_model(self, filepath: str = "learning_model.pkl"):
        """Load the learning model from file"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)

                self.q_table = model_data.get('q_table', {})
                self.episode_count = model_data.get('episode_count', 0)
                self.total_reward = model_data.get('total_reward', 0.0)
                self.average_performance = model_data.get('average_performance', 0.0)

                self.get_logger().info(f"Model loaded from {filepath}")
            except Exception as e:
                self.get_logger().error(f"Error loading model: {e}")
        else:
            self.get_logger().info("No saved model found, starting fresh")

    def get_learning_stats(self) -> Dict[str, float]:
        """Get learning statistics"""
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'buffer_size': len(self.experience_buffer),
            'average_performance': self.average_performance,
            'total_reward': self.total_reward,
            'enabled': self.learning_enabled
        }

    def get_performance_history(self) -> List[float]:
        """Get performance history"""
        return self.performance_history.copy()


def main(args=None):
    rclpy.init(args=args)

    learning_node = LearningNode()

    try:
        rclpy.spin(learning_node)
    except KeyboardInterrupt:
        # Save model before shutdown
        learning_node.save_model()
        pass
    finally:
        learning_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()