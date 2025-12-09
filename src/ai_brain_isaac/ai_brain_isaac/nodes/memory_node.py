#!/usr/bin/env python3

"""
Memory Node for Humanoid Robot AI Brain

This node handles memory management, knowledge representation, and
information storage similar to how memory works in biological systems.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Header
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Pose, Point, Vector3
from nav_msgs.msg import Odometry
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import threading
from dataclasses import dataclass
import json
import os
from enum import Enum


class MemoryType(Enum):
    EPISODIC = "episodic"      # Memory of specific events
    SEMANTIC = "semantic"      # General knowledge and facts
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"        # Short-term memory


@dataclass
class MemoryItem:
    """Represents a memory item with metadata"""
    id: str
    type: MemoryType
    content: Any
    timestamp: float
    importance: float  # 0.0 to 1.0
    tags: List[str]
    decay_rate: float  # How quickly memory fades


class MemoryNode(Node):
    def __init__(self):
        super().__init__('memory_node')

        # Publishers for memory operations
        self.memory_status_publisher = self.create_publisher(String, 'ai/memory_status', 10)
        self.recall_publisher = self.create_publisher(String, 'ai/memory_recall', 10)

        # Subscribers for information to store
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.user_input_subscriber = self.create_subscription(
            String, 'ai/user_input', self.user_input_callback, 10)
        self.memory_request_subscriber = self.create_subscription(
            String, 'ai/memory_request', self.memory_request_callback, 10)

        # Timer for memory management
        self.memory_timer = self.create_timer(1.0, self.memory_management_callback)  # 1 Hz

        # Memory storage
        self.episodic_memory = {}  # Event memories
        self.semantic_memory = {}  # Factual knowledge
        self.procedural_memory = {}  # Skills and procedures
        self.working_memory = {}  # Short-term memory

        # Memory parameters
        self.memory_enabled = True
        self.working_memory_size = 10  # Max items in working memory
        self.long_term_memory_size = 1000  # Max items in long-term memory per type
        self.forgetting_threshold = 0.1  # Importance threshold for forgetting
        self.decay_rate = 0.001  # Rate at which memories fade
        self.sim_time = time.time()

        # Threading lock for memory access
        self.memory_lock = threading.Lock()

        # Load any saved memory
        self.load_memory()

        self.get_logger().info('Memory Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state data - store as episodic memory"""
        with self.memory_lock:
            if self.memory_enabled:
                # Create an episodic memory of the joint state
                content = {
                    'type': 'joint_state',
                    'timestamp': time.time(),
                    'position': [p for p in msg.position],
                    'velocity': [v for v in msg.velocity],
                    'effort': [e for e in msg.effort],
                    'joint_names': [n for n in msg.name]
                }

                memory_item = MemoryItem(
                    id=f"joint_state_{time.time()}",
                    type=MemoryType.EPISODIC,
                    content=content,
                    timestamp=time.time(),
                    importance=0.5,  # Medium importance
                    tags=['joint_state', 'sensor_data'],
                    decay_rate=0.002
                )

                self.store_memory(memory_item)

    def odom_callback(self, msg):
        """Callback for odometry data - store as episodic memory"""
        with self.memory_lock:
            if self.memory_enabled:
                # Create an episodic memory of the location
                content = {
                    'type': 'location',
                    'position': {
                        'x': msg.pose.pose.position.x,
                        'y': msg.pose.pose.position.y,
                        'z': msg.pose.pose.position.z
                    },
                    'orientation': {
                        'x': msg.pose.pose.orientation.x,
                        'y': msg.pose.pose.orientation.y,
                        'z': msg.pose.pose.orientation.z,
                        'w': msg.pose.pose.orientation.w
                    },
                    'velocity': {
                        'linear': {
                            'x': msg.twist.twist.linear.x,
                            'y': msg.twist.twist.linear.y,
                            'z': msg.twist.twist.linear.z
                        },
                        'angular': {
                            'x': msg.twist.twist.angular.x,
                            'y': msg.twist.twist.angular.y,
                            'z': msg.twist.twist.angular.z
                        }
                    }
                }

                memory_item = MemoryItem(
                    id=f"location_{time.time()}",
                    type=MemoryType.EPISODIC,
                    content=content,
                    timestamp=time.time(),
                    importance=0.7,  # Higher importance for location
                    tags=['location', 'navigation', 'pose'],
                    decay_rate=0.001
                )

                self.store_memory(memory_item)

    def imu_callback(self, msg):
        """Callback for IMU data - store as episodic memory"""
        with self.memory_lock:
            if self.memory_enabled:
                content = {
                    'type': 'imu_data',
                    'linear_acceleration': {
                        'x': msg.linear_acceleration.x,
                        'y': msg.linear_acceleration.y,
                        'z': msg.linear_acceleration.z
                    },
                    'angular_velocity': {
                        'x': msg.angular_velocity.x,
                        'y': msg.angular_velocity.y,
                        'z': msg.angular_velocity.z
                    }
                }

                memory_item = MemoryItem(
                    id=f"imu_{time.time()}",
                    type=MemoryType.EPISODIC,
                    content=content,
                    timestamp=time.time(),
                    importance=0.4,
                    tags=['imu', 'balance', 'acceleration'],
                    decay_rate=0.003
                )

                self.store_memory(memory_item)

    def laser_callback(self, msg):
        """Callback for laser scan data - store as episodic memory"""
        with self.memory_lock:
            if self.memory_enabled:
                content = {
                    'type': 'laser_scan',
                    'ranges': [r for r in msg.ranges if not math.isnan(r)],
                    'intensities': [i for i in msg.intensities],
                    'angle_min': msg.angle_min,
                    'angle_max': msg.angle_max,
                    'angle_increment': msg.angle_increment
                }

                # Calculate importance based on obstacle detection
                min_range = min(content['ranges']) if content['ranges'] else float('inf')
                importance = 0.9 if min_range < 1.0 else 0.3  # High importance if close obstacles

                memory_item = MemoryItem(
                    id=f"laser_{time.time()}",
                    type=MemoryType.EPISODIC,
                    content=content,
                    timestamp=time.time(),
                    importance=importance,
                    tags=['laser', 'obstacle_detection', 'navigation'],
                    decay_rate=0.002
                )

                self.store_memory(memory_item)

    def user_input_callback(self, msg):
        """Callback for user input - store as semantic memory"""
        with self.memory_lock:
            if self.memory_enabled:
                # Store user input as semantic knowledge
                content = {
                    'type': 'user_command',
                    'text': msg.data,
                    'timestamp': time.time()
                }

                # Determine importance based on keywords
                importance = 0.3
                if any(word in msg.data.lower() for word in ['help', 'emergency', 'stop', 'danger']):
                    importance = 0.9
                elif any(word in msg.data.lower() for word in ['remember', 'learn', 'teach']):
                    importance = 0.7

                memory_item = MemoryItem(
                    id=f"user_input_{time.time()}",
                    type=MemoryType.SEMANTIC,
                    content=content,
                    timestamp=time.time(),
                    importance=importance,
                    tags=['user_input', 'command', 'instruction'],
                    decay_rate=0.0005  # Slower decay for important instructions
                )

                self.store_memory(memory_item)

    def memory_request_callback(self, msg):
        """Callback for memory requests"""
        with self.memory_lock:
            request = msg.data.lower()

            if request.startswith('recall:'):
                # Extract query
                query = request[7:].strip()  # Remove 'recall:' prefix
                results = self.recall_memory(query)

                # Publish recall results
                recall_msg = String()
                recall_msg.data = f"RECALL: Query='{query}', Results={len(results)}"
                self.recall_publisher.publish(recall_msg)

    def store_memory(self, memory_item: MemoryItem):
        """Store a memory item in the appropriate memory system"""
        # Add to working memory first
        self.working_memory[memory_item.id] = memory_item

        # Trim working memory if too large
        if len(self.working_memory) > self.working_memory_size:
            # Remove oldest items
            oldest_id = min(self.working_memory.keys(), key=lambda k: self.working_memory[k].timestamp)
            del self.working_memory[oldest_id]

        # Add to long-term memory based on type
        memory_dict = self.get_memory_dict_by_type(memory_item.type)
        memory_dict[memory_item.id] = memory_item

        # Trim long-term memory if too large
        if len(memory_dict) > self.long_term_memory_size:
            # Remove items with lowest importance
            sorted_items = sorted(memory_dict.items(), key=lambda x: x[1].importance)
            for item_id, _ in sorted_items[:len(memory_dict) - self.long_term_memory_size]:
                del memory_dict[item_id]

    def get_memory_dict_by_type(self, memory_type: MemoryType):
        """Get the appropriate memory dictionary based on type"""
        if memory_type == MemoryType.EPISODIC:
            return self.episodic_memory
        elif memory_type == MemoryType.SEMANTIC:
            return self.semantic_memory
        elif memory_type == MemoryType.PROCEDURAL:
            return self.procedural_memory
        else:  # Working memory is handled separately
            return self.episodic_memory

    def recall_memory(self, query: str) -> List[MemoryItem]:
        """Recall memories matching the query"""
        results = []

        # Search in all memory types
        all_memories = {**self.episodic_memory, **self.semantic_memory,
                       **self.procedural_memory, **self.working_memory}

        for memory_id, memory_item in all_memories.items():
            # Simple keyword matching for now
            content_str = json.dumps(memory_item.content, default=str).lower()
            if query.lower() in content_str or any(tag in query.lower() for tag in memory_item.tags):
                results.append(memory_item)

        # Sort by importance and recency
        results.sort(key=lambda x: x.importance * (1 + 1/(time.time() - x.timestamp + 1)), reverse=True)

        return results[:10]  # Return top 10 matches

    def memory_management_callback(self):
        """Main memory management callback - handle forgetting and decay"""
        if not self.memory_enabled:
            return

        current_time = time.time()

        with self.memory_lock:
            # Apply decay to all memories
            self.apply_memory_decay(current_time)

            # Remove low-importance memories
            self.forget_memories()

        # Publish memory status
        total_memories = (len(self.episodic_memory) + len(self.semantic_memory) +
                         len(self.procedural_memory) + len(self.working_memory))

        status_msg = String()
        status_msg.data = f"MEMORY: Total={total_memories}, Episodic={len(self.episodic_memory)}, Semantic={len(self.semantic_memory)}, Time={current_time:.2f}"
        self.memory_status_publisher.publish(status_msg)

    def apply_memory_decay(self, current_time: float):
        """Apply decay to memory importance over time"""
        all_memories = [self.episodic_memory, self.semantic_memory,
                       self.procedural_memory, self.working_memory]

        for memory_dict in all_memories:
            for memory_id, memory_item in list(memory_dict.items()):
                # Apply decay based on time elapsed and decay rate
                time_elapsed = current_time - memory_item.timestamp
                decay_factor = math.exp(-memory_item.decay_rate * time_elapsed)
                memory_item.importance *= decay_factor

                # If importance drops below threshold, mark for removal
                if memory_item.importance < 0.01:
                    del memory_dict[memory_id]

    def forget_memories(self):
        """Remove low-importance memories to manage memory size"""
        all_memories = [self.episodic_memory, self.semantic_memory, self.procedural_memory]

        for memory_dict in all_memories:
            if len(memory_dict) > self.long_term_memory_size * 0.8:  # Start forgetting when 80% full
                # Remove memories below threshold
                to_remove = [id for id, item in memory_dict.items()
                            if item.importance < self.forgetting_threshold]

                for memory_id in to_remove:
                    del memory_dict[memory_id]

    def store_procedural_memory(self, skill_name: str, procedure: Dict[str, Any], tags: List[str] = None):
        """Store a procedural memory (skill or procedure)"""
        if tags is None:
            tags = ['skill', 'procedure']

        content = {
            'type': 'procedure',
            'name': skill_name,
            'procedure': procedure,
            'timestamp': time.time()
        }

        memory_item = MemoryItem(
            id=f"skill_{skill_name}_{time.time()}",
            type=MemoryType.PROCEDURAL,
            content=content,
            timestamp=time.time(),
            importance=0.8,  # Skills are generally important
            tags=tags,
            decay_rate=0.0001  # Very slow decay for skills
        )

        with self.memory_lock:
            self.store_memory(memory_item)

    def recall_procedural_memory(self, skill_name: str) -> Optional[Dict]:
        """Recall a specific procedural memory"""
        with self.memory_lock:
            for memory_id, memory_item in self.procedural_memory.items():
                if (memory_item.type == MemoryType.PROCEDURAL and
                    memory_item.content.get('name') == skill_name):
                    return memory_item.content.get('procedure')

        return None

    def store_semantic_fact(self, fact: str, tags: List[str] = None):
        """Store a semantic fact"""
        if tags is None:
            tags = ['fact', 'knowledge']

        content = {
            'type': 'fact',
            'text': fact,
            'timestamp': time.time()
        }

        memory_item = MemoryItem(
            id=f"fact_{time.time()}",
            type=MemoryType.SEMANTIC,
            content=content,
            timestamp=time.time(),
            importance=0.6,  # Facts are moderately important
            tags=tags,
            decay_rate=0.0002
        )

        with self.memory_lock:
            self.store_memory(memory_item)

    def save_memory(self, filepath: str = "memory.json"):
        """Save memory to file"""
        memory_data = {
            'episodic_memory': {k: {
                'id': v.id,
                'type': v.type.value,
                'content': v.content,
                'timestamp': v.timestamp,
                'importance': v.importance,
                'tags': v.tags,
                'decay_rate': v.decay_rate
            } for k, v in self.episodic_memory.items()},
            'semantic_memory': {k: {
                'id': v.id,
                'type': v.type.value,
                'content': v.content,
                'timestamp': v.timestamp,
                'importance': v.importance,
                'tags': v.tags,
                'decay_rate': v.decay_rate
            } for k, v in self.semantic_memory.items()},
            'procedural_memory': {k: {
                'id': v.id,
                'type': v.type.value,
                'content': v.content,
                'timestamp': v.timestamp,
                'importance': v.importance,
                'tags': v.tags,
                'decay_rate': v.decay_rate
            } for k, v in self.procedural_memory.items()},
            'working_memory': {k: {
                'id': v.id,
                'type': v.type.value,
                'content': v.content,
                'timestamp': v.timestamp,
                'importance': v.importance,
                'tags': v.tags,
                'decay_rate': v.decay_rate
            } for k, v in self.working_memory.items()},
            'metadata': {
                'saved_at': time.time(),
                'total_memories': len(self.episodic_memory) + len(self.semantic_memory) +
                                 len(self.procedural_memory) + len(self.working_memory)
            }
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(memory_data, f, indent=2)
            self.get_logger().info(f"Memory saved to {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error saving memory: {e}")

    def load_memory(self, filepath: str = "memory.json"):
        """Load memory from file"""
        if not os.path.exists(filepath):
            self.get_logger().info("No saved memory found, starting fresh")
            return

        try:
            with open(filepath, 'r') as f:
                memory_data = json.load(f)

            # Load episodic memory
            for k, v in memory_data.get('episodic_memory', {}).items():
                memory_item = MemoryItem(
                    id=v['id'],
                    type=MemoryType(v['type']),
                    content=v['content'],
                    timestamp=v['timestamp'],
                    importance=v['importance'],
                    tags=v['tags'],
                    decay_rate=v['decay_rate']
                )
                self.episodic_memory[k] = memory_item

            # Load semantic memory
            for k, v in memory_data.get('semantic_memory', {}).items():
                memory_item = MemoryItem(
                    id=v['id'],
                    type=MemoryType(v['type']),
                    content=v['content'],
                    timestamp=v['timestamp'],
                    importance=v['importance'],
                    tags=v['tags'],
                    decay_rate=v['decay_rate']
                )
                self.semantic_memory[k] = memory_item

            # Load procedural memory
            for k, v in memory_data.get('procedural_memory', {}).items():
                memory_item = MemoryItem(
                    id=v['id'],
                    type=MemoryType(v['type']),
                    content=v['content'],
                    timestamp=v['timestamp'],
                    importance=v['importance'],
                    tags=v['tags'],
                    decay_rate=v['decay_rate']
                )
                self.procedural_memory[k] = memory_item

            # Load working memory
            for k, v in memory_data.get('working_memory', {}).items():
                memory_item = MemoryItem(
                    id=v['id'],
                    type=MemoryType(v['type']),
                    content=v['content'],
                    timestamp=v['timestamp'],
                    importance=v['importance'],
                    tags=v['tags'],
                    decay_rate=v['decay_rate']
                )
                self.working_memory[k] = memory_item

            self.get_logger().info(f"Memory loaded from {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error loading memory: {e}")

    def enable_memory(self, enable: bool):
        """Enable or disable memory operations"""
        self.memory_enabled = enable
        self.get_logger().info(f"Memory operations {'enabled' if enable else 'disabled'}")

    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics"""
        return {
            'episodic_count': len(self.episodic_memory),
            'semantic_count': len(self.semantic_memory),
            'procedural_count': len(self.procedural_memory),
            'working_count': len(self.working_memory),
            'total': len(self.episodic_memory) + len(self.semantic_memory) +
                     len(self.procedural_memory) + len(self.working_memory),
            'enabled': self.memory_enabled
        }

    def clear_memory(self, memory_type: Optional[MemoryType] = None):
        """Clear memory, optionally by type"""
        with self.memory_lock:
            if memory_type is None:
                # Clear all memory
                self.episodic_memory.clear()
                self.semantic_memory.clear()
                self.procedural_memory.clear()
                self.working_memory.clear()
                self.get_logger().info("All memory cleared")
            else:
                # Clear specific memory type
                if memory_type == MemoryType.EPISODIC:
                    self.episodic_memory.clear()
                elif memory_type == MemoryType.SEMANTIC:
                    self.semantic_memory.clear()
                elif memory_type == MemoryType.PROCEDURAL:
                    self.procedural_memory.clear()
                elif memory_type == MemoryType.WORKING:
                    self.working_memory.clear()

                self.get_logger().info(f"{memory_type.value} memory cleared")


def main(args=None):
    rclpy.init(args=args)

    memory_node = MemoryNode()

    try:
        rclpy.spin(memory_node)
    except KeyboardInterrupt:
        # Save memory before shutdown
        memory_node.save_memory()
        pass
    finally:
        memory_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()