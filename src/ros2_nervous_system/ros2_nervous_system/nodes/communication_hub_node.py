#!/usr/bin/env python3

"""
Communication Hub Node for Humanoid Robot

This node serves as a communication hub that manages message routing
and coordination between different subsystems of the humanoid robot,
similar to how neural pathways connect different brain regions.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
import time
from collections import defaultdict, deque
import threading


class CommunicationHubNode(Node):
    def __init__(self):
        super().__init__('communication_hub_node')

        # Publishers for system-wide communication
        self.system_status_publisher = self.create_publisher(String, 'system_status', 10)
        self.message_log_publisher = self.create_publisher(String, 'message_log', 10)
        self.health_report_publisher = self.create_publisher(String, 'health_report', 10)

        # Subscribers for monitoring all system messages
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.motion_command_subscriber = self.create_subscription(
            Twist, 'motion_commands', self.motion_command_callback, 10)
        self.behavior_state_subscriber = self.create_subscription(
            String, 'behavior_state', self.behavior_state_callback, 10)
        self.system_status_subscriber = self.create_subscription(
            String, 'system_status', self.system_status_callback, 10)

        # Timer for system monitoring and health checks
        self.monitor_timer = self.create_timer(0.5, self.system_monitor_callback)  # 2 Hz

        # Communication tracking
        self.message_history = deque(maxlen=100)  # Store last 100 messages
        self.node_health = defaultdict(dict)      # Track health of each node
        self.message_stats = defaultdict(int)     # Count messages by type
        self.last_message_time = defaultdict(float)  # Track when each node last communicated
        self.communication_matrix = defaultdict(lambda: defaultdict(bool))  # Track node connections

        # System status
        self.system_uptime = time.time()
        self.active_nodes = set()
        self.message_queue = []
        self.message_queue_lock = threading.Lock()

        self.get_logger().info('Communication Hub Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state messages"""
        self.log_message('joint_states', msg)
        self.update_node_health('joint_state_publisher', time.time())

    def motion_command_callback(self, msg):
        """Callback for motion command messages"""
        self.log_message('motion_commands', msg)
        self.update_node_health('motion_command_publisher', time.time())

    def behavior_state_callback(self, msg):
        """Callback for behavior state messages"""
        self.log_message('behavior_state', msg)
        self.update_node_health('behavior_manager', time.time())

    def system_status_callback(self, msg):
        """Callback for system status messages"""
        self.log_message('system_status', msg)
        # Extract node name from status message if available
        if ':' in msg.data:
            node_name = msg.data.split(':')[0].split('_')[0]  # Simplified extraction
            self.update_node_health(node_name, time.time())

    def log_message(self, topic, msg):
        """Log incoming messages with timestamp"""
        timestamp = time.time()
        message_info = {
            'topic': topic,
            'timestamp': timestamp,
            'size': len(str(msg)) if str(msg) else 0,  # Approximate size
            'source': self.extract_source(topic)  # Simplified source extraction
        }

        with self.message_queue_lock:
            self.message_history.append(message_info)
            self.message_stats[topic] += 1

        # Log significant messages
        if topic in ['system_status', 'behavior_state']:
            self.get_logger().debug(f'Message on {topic}: {str(msg)[:100]}...')

    def extract_source(self, topic):
        """Extract source node name from topic"""
        # Simplified source extraction - in reality would use more sophisticated methods
        if 'joint' in topic:
            return 'sensor_fusion'
        elif 'motion' in topic:
            return 'behavior_manager'
        elif 'behavior' in topic:
            return 'behavior_manager'
        else:
            return 'unknown'

    def update_node_health(self, node_name, timestamp):
        """Update health status of a node"""
        self.active_nodes.add(node_name)
        self.last_message_time[node_name] = timestamp

        # Update health metrics
        if node_name not in self.node_health:
            self.node_health[node_name] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'message_count': 0,
                'status': 'HEALTHY'
            }
        else:
            self.node_health[node_name]['last_seen'] = timestamp
            self.node_health[node_name]['message_count'] += 1

        # Check if node is responsive (no message in 5 seconds = WARNING)
        if timestamp - self.last_message_time[node_name] > 5.0:
            self.node_health[node_name]['status'] = 'UNRESPONSIVE'
        else:
            self.node_health[node_name]['status'] = 'HEALTHY'

    def system_monitor_callback(self):
        """Timer callback for system monitoring"""
        current_time = time.time()

        # Check for unresponsive nodes
        for node_name in list(self.active_nodes):
            if current_time - self.last_message_time[node_name] > 5.0:
                self.get_logger().warn(f'Node {node_name} is unresponsive')
                self.node_health[node_name]['status'] = 'UNRESPONSIVE'

        # Publish system status
        status_msg = String()
        status_msg.data = f"SYSTEM_OK: {len(self.active_nodes)} nodes active, uptime: {current_time - self.system_uptime:.1f}s"
        self.system_status_publisher.publish(status_msg)

        # Publish health report periodically
        if int(current_time) % 10 == 0:  # Every 10 seconds
            self.publish_health_report()

    def publish_health_report(self):
        """Publish comprehensive health report"""
        report_parts = []
        report_parts.append(f"Health Report - {time.strftime('%H:%M:%S')}")
        report_parts.append(f"Active Nodes: {len(self.active_nodes)}")

        for node_name, health_info in self.node_health.items():
            status = health_info['status']
            msg_count = health_info['message_count']
            report_parts.append(f"  {node_name}: {status} ({msg_count} msgs)")

        report_parts.append(f"Total Messages: {sum(self.message_stats.values())}")
        report_parts.append(f"Message Types: {dict(self.message_stats)}")

        report_msg = String()
        report_msg.data = " | ".join(report_parts)
        self.health_report_publisher.publish(report_msg)

    def route_message(self, topic, message, priority='normal'):
        """Route a message to appropriate subscribers"""
        # This would contain more sophisticated routing logic in a real system
        # For now, we just log the routing action
        self.get_logger().debug(f'Routing message to {topic} with priority {priority}')

    def get_network_topology(self):
        """Get current network topology"""
        topology = {
            'active_nodes': list(self.active_nodes),
            'connections': dict(self.communication_matrix),
            'message_stats': dict(self.message_stats),
            'health_status': {node: info['status'] for node, info in self.node_health.items()}
        }
        return topology

    def get_message_history(self, limit=10):
        """Get recent message history"""
        with self.message_queue_lock:
            recent_messages = list(self.message_history)[-limit:]
        return recent_messages

    def reset_communication_stats(self):
        """Reset all communication statistics"""
        self.message_stats.clear()
        self.message_history.clear()
        for node in self.node_health:
            self.node_health[node]['message_count'] = 0


def main(args=None):
    rclpy.init(args=args)

    communication_hub_node = CommunicationHubNode()

    try:
        rclpy.spin(communication_hub_node)
    except KeyboardInterrupt:
        pass
    finally:
        communication_hub_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()