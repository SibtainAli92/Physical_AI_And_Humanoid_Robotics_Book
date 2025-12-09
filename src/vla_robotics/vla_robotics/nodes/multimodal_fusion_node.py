#!/usr/bin/env python3

"""
Multimodal Fusion Node for Humanoid Robot VLA System

This node integrates information from vision, language, and action systems
to create a unified understanding and coordinate multimodal behaviors.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Int32
from sensor_msgs.msg import Image, JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose, Point, Vector3
from nav_msgs.msg import Odometry
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import threading
from dataclasses import dataclass
from enum import Enum
import json


class FusionState(Enum):
    INACTIVE = "inactive"
    VISION_PROCESSING = "vision_processing"
    LANGUAGE_PROCESSING = "language_processing"
    ACTION_PLANNING = "action_planning"
    FUSION_ACTIVE = "fusion_active"
    EXECUTION = "execution"


@dataclass
class MultimodalInput:
    """Represents a multimodal input with timestamp and confidence"""
    modality: str  # 'vision', 'language', 'action', 'other'
    data: Any
    timestamp: float
    confidence: float
    source: str


@dataclass
class FusedRepresentation:
    """Represents the fused multimodal understanding"""
    timestamp: float
    visual_features: Dict[str, Any]
    linguistic_features: Dict[str, Any]
    action_features: Dict[str, Any]
    fused_context: Dict[str, Any]
    overall_confidence: float


class MultimodalFusionNode(Node):
    def __init__(self):
        super().__init__('multimodal_fusion_node')

        # Publishers for fused output
        self.fused_output_publisher = self.create_publisher(String, 'vla/fused_output', 10)
        self.fusion_status_publisher = self.create_publisher(String, 'vla/fusion_status', 10)
        self.command_output_publisher = self.create_publisher(String, 'vla/fusion_command', 10)
        self.attention_map_publisher = self.create_publisher(String, 'vla/attention_map', 10)

        # Subscribers for multimodal inputs
        self.vision_input_subscriber = self.create_subscription(
            String, 'vla/vision_input', self.vision_input_callback, 10)
        self.language_input_subscriber = self.create_subscription(
            String, 'vla/language_input', self.language_input_callback, 10)
        self.action_input_subscriber = self.create_subscription(
            String, 'vla/action_input', self.action_input_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Timer for multimodal fusion
        self.fusion_timer = self.create_timer(0.1, self.multimodal_fusion_callback)  # 10 Hz

        # Multimodal fusion state
        self.fusion_state = FusionState.INACTIVE
        self.multimodal_inputs = []
        self.fused_representations = []
        self.current_fused_state: Optional[FusedRepresentation] = None
        self.multimodal_fusion_enabled = True
        self.sim_time = time.time()
        self.last_fusion_time = time.time()

        # Fusion parameters
        self.max_input_buffer_size = 50
        self.fusion_window_size = 5  # Number of inputs to consider for fusion
        self.temporal_decay_rate = 0.95  # Rate at which older inputs lose relevance
        self.confidence_threshold = 0.3  # Minimum confidence for consideration

        # Cross-modal attention weights
        self.attention_weights = {
            'vision_to_language': 0.7,
            'language_to_vision': 0.8,
            'vision_to_action': 0.6,
            'language_to_action': 0.9,
            'action_to_vision': 0.5,
            'action_to_language': 0.4
        }

        # State tracking
        self.current_pose = Pose()
        self.current_joint_states = {}
        self.scene_context = {}
        self.language_context = {}
        self.action_context = {}

        # Threading lock for data access
        self.data_lock = threading.Lock()

        self.get_logger().info('Multimodal Fusion Node initialized')

    def vision_input_callback(self, msg):
        """Callback for vision inputs"""
        with self.data_lock:
            try:
                # Parse vision input message
                vision_data = json.loads(msg.data)
                confidence = vision_data.get('confidence', 0.8)

                input_entry = MultimodalInput(
                    modality='vision',
                    data=vision_data,
                    timestamp=time.time(),
                    confidence=confidence,
                    source='vision_processor'
                )

                self.add_multimodal_input(input_entry)
                self.get_logger().debug(f'Added vision input: {vision_data.get("type", "unknown")}')
            except json.JSONDecodeError:
                # If not JSON, treat as simple string
                input_entry = MultimodalInput(
                    modality='vision',
                    data={'type': 'simple_vision', 'content': msg.data},
                    timestamp=time.time(),
                    confidence=0.7,
                    source='vision_processor'
                )
                self.add_multimodal_input(input_entry)

    def language_input_callback(self, msg):
        """Callback for language inputs"""
        with self.data_lock:
            # Process language input
            input_entry = MultimodalInput(
                modality='language',
                data={'type': 'command', 'text': msg.data},
                timestamp=time.time(),
                confidence=0.9,  # Language input typically has high confidence
                source='language_processor'
            )

            self.add_multimodal_input(input_entry)
            self.language_context['last_command'] = msg.data
            self.get_logger().info(f'Added language input: {msg.data}')

    def action_input_callback(self, msg):
        """Callback for action inputs"""
        with self.data_lock:
            # Process action input
            input_entry = MultimodalInput(
                modality='action',
                data={'type': 'action', 'content': msg.data},
                timestamp=time.time(),
                confidence=1.0,  # Action input confidence is high when executing
                source='action_processor'
            )

            self.add_multimodal_input(input_entry)
            self.get_logger().debug(f'Added action input: {msg.data}')

    def odom_callback(self, msg):
        """Callback for odometry data"""
        with self.data_lock:
            self.current_pose = msg.pose.pose

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        with self.data_lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_joint_states[name] = msg.position[i]

    def add_multimodal_input(self, input_entry: MultimodalInput):
        """Add a multimodal input to the buffer"""
        self.multimodal_inputs.append(input_entry)

        # Trim buffer if too large
        if len(self.multimodal_inputs) > self.max_input_buffer_size:
            self.multimodal_inputs.pop(0)

    def multimodal_fusion_callback(self):
        """Main multimodal fusion callback"""
        if not self.multimodal_fusion_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_fusion_time
        self.last_fusion_time = current_time

        with self.data_lock:
            # Perform multimodal fusion
            fused_state = self.perform_fusion()

            if fused_state:
                self.current_fused_state = fused_state
                self.fused_representations.append(fused_state)

                # Trim fused representations buffer
                if len(self.fused_representations) > self.max_input_buffer_size:
                    self.fused_representations.pop(0)

                # Generate appropriate output based on fused state
                self.generate_fusion_output(fused_state)

                # Update fusion state based on current inputs
                self.update_fusion_state()

        # Publish fusion status
        status_msg = String()
        status_msg.data = f"FUSION: State={self.fusion_state.value}, Inputs={len(self.multimodal_inputs)}, Fused={len(self.fused_representations)}, Time={current_time:.2f}"
        self.fusion_status_publisher.publish(status_msg)

        # Publish attention map
        self.publish_attention_map()

    def perform_fusion(self) -> Optional[FusedRepresentation]:
        """Perform multimodal fusion on recent inputs"""
        if not self.multimodal_inputs:
            return None

        # Get recent inputs within fusion window
        recent_inputs = self.get_recent_inputs(self.fusion_window_size)
        if not recent_inputs:
            return None

        # Separate inputs by modality
        vision_inputs = [inp for inp in recent_inputs if inp.modality == 'vision']
        language_inputs = [inp for inp in recent_inputs if inp.modality == 'language']
        action_inputs = [inp for inp in recent_inputs if inp.modality == 'action']

        # Extract features from each modality
        visual_features = self.extract_visual_features(vision_inputs)
        linguistic_features = self.extract_linguistic_features(language_inputs)
        action_features = self.extract_action_features(action_inputs)

        # Perform cross-modal attention and fusion
        fused_context = self.compute_fused_context(visual_features, linguistic_features, action_features)

        # Calculate overall confidence based on input confidences
        total_confidence = sum(inp.confidence for inp in recent_inputs) / len(recent_inputs)

        fused_repr = FusedRepresentation(
            timestamp=time.time(),
            visual_features=visual_features,
            linguistic_features=linguistic_features,
            action_features=action_features,
            fused_context=fused_context,
            overall_confidence=total_confidence
        )

        return fused_repr

    def get_recent_inputs(self, count: int) -> List[MultimodalInput]:
        """Get the most recent multimodal inputs"""
        # Sort inputs by timestamp and return the most recent ones
        sorted_inputs = sorted(self.multimodal_inputs, key=lambda x: x.timestamp, reverse=True)
        return sorted_inputs[:min(count, len(sorted_inputs))]

    def extract_visual_features(self, vision_inputs: List[MultimodalInput]) -> Dict[str, Any]:
        """Extract features from vision inputs"""
        features = {
            'objects': [],
            'locations': [],
            'colors': [],
            'sizes': [],
            'spatial_relations': [],
            'scene_type': 'unknown',
            'confidence': 0.0
        }

        if not vision_inputs:
            return features

        # Aggregate visual information
        for v_input in vision_inputs:
            data = v_input.data
            if isinstance(data, dict):
                if 'objects' in data:
                    features['objects'].extend(data['objects'])
                if 'location' in data:
                    features['locations'].append(data['location'])
                if 'color' in data:
                    features['colors'].append(data['color'])
                if 'size' in data:
                    features['sizes'].append(data['size'])
                if 'spatial_relation' in data:
                    features['spatial_relations'].append(data['spatial_relation'])

        # Calculate aggregate confidence
        if vision_inputs:
            features['confidence'] = sum(v.confidence for v in vision_inputs) / len(vision_inputs)

        return features

    def extract_linguistic_features(self, language_inputs: List[MultimodalInput]) -> Dict[str, Any]:
        """Extract features from language inputs"""
        features = {
            'commands': [],
            'entities': [],
            'actions': [],
            'spatial_refs': [],
            'descriptions': [],
            'intent': 'unknown',
            'confidence': 0.0
        }

        if not language_inputs:
            return features

        # Aggregate linguistic information
        for l_input in language_inputs:
            data = l_input.data
            if isinstance(data, dict):
                text = data.get('text', '')
            else:
                text = str(l_input.data)

            # Extract linguistic features from text
            features['commands'].append(text)

            # Simple entity extraction
            entities = self.extract_entities_from_text(text)
            features['entities'].extend(entities)

            # Simple action extraction
            actions = self.extract_actions_from_text(text)
            features['actions'].extend(actions)

        # Calculate aggregate confidence
        if language_inputs:
            features['confidence'] = sum(l.confidence for l in language_inputs) / len(language_inputs)

        return features

    def extract_action_features(self, action_inputs: List[MultimodalInput]) -> Dict[str, Any]:
        """Extract features from action inputs"""
        features = {
            'executed_actions': [],
            'action_outcomes': [],
            'action_success': [],
            'next_actions': [],
            'confidence': 0.0
        }

        if not action_inputs:
            return features

        # Aggregate action information
        for a_input in action_inputs:
            data = a_input.data
            if isinstance(data, dict):
                features['executed_actions'].append(data)
            else:
                features['executed_actions'].append({'action': str(data)})

        # Calculate aggregate confidence
        if action_inputs:
            features['confidence'] = sum(a.confidence for a in action_inputs) / len(action_inputs)

        return features

    def extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text (simplified)"""
        # Simple keyword-based entity extraction
        entities = []
        entity_keywords = [
            'object', 'person', 'table', 'chair', 'cup', 'box', 'ball',
            'red', 'blue', 'green', 'yellow', 'big', 'small', 'left', 'right'
        ]

        text_lower = text.lower()
        for keyword in entity_keywords:
            if keyword in text_lower:
                entities.append(keyword)

        return entities

    def extract_actions_from_text(self, text: str) -> List[str]:
        """Extract actions from text (simplified)"""
        # Simple keyword-based action extraction
        actions = []
        action_keywords = [
            'go', 'move', 'walk', 'grasp', 'pick', 'take', 'put', 'place',
            'point', 'show', 'look', 'turn', 'stop', 'wait'
        ]

        text_lower = text.lower()
        for keyword in action_keywords:
            if keyword in text_lower:
                actions.append(keyword)

        return actions

    def compute_fused_context(self, visual_features: Dict, linguistic_features: Dict, action_features: Dict) -> Dict[str, Any]:
        """Compute the fused context by combining features from all modalities"""
        fused_context = {
            'scene_understanding': {},
            'command_interpretation': {},
            'action_recommendation': {},
            'attention_focus': {},
            'confidence_alignment': {}
        }

        # Scene understanding - combine visual and linguistic info
        fused_context['scene_understanding'] = {
            'detected_objects': visual_features.get('objects', []),
            'mentioned_entities': linguistic_features.get('entities', []),
            'object_entity_matches': self.match_objects_to_entities(
                visual_features.get('objects', []),
                linguistic_features.get('entities', [])
            )
        }

        # Command interpretation - ground language in visual context
        fused_context['command_interpretation'] = {
            'intended_action': linguistic_features.get('actions', []),
            'target_objects': self.identify_target_objects(
                linguistic_features.get('entities', []),
                visual_features.get('objects', [])
            ),
            'spatial_constraints': linguistic_features.get('spatial_refs', []) + visual_features.get('spatial_relations', [])
        }

        # Action recommendation - suggest next actions
        fused_context['action_recommendation'] = {
            'recommended_action': self.recommend_action(
                fused_context['command_interpretation'],
                action_features.get('executed_actions', [])
            )
        }

        # Attention focus - determine where to focus processing
        fused_context['attention_focus'] = {
            'visual_attention': self.compute_visual_attention(visual_features, linguistic_features),
            'linguistic_attention': self.compute_linguistic_attention(linguistic_features, visual_features)
        }

        # Confidence alignment - ensure consistent confidence across modalities
        fused_context['confidence_alignment'] = {
            'visual_confidence': visual_features.get('confidence', 0.0),
            'linguistic_confidence': linguistic_features.get('confidence', 0.0),
            'action_confidence': action_features.get('confidence', 0.0),
            'overall_confidence': (
                visual_features.get('confidence', 0.0) * self.attention_weights['vision_to_language'] +
                linguistic_features.get('confidence', 0.0) * self.attention_weights['language_to_vision'] +
                action_features.get('confidence', 0.0) * self.attention_weights['action_to_vision']
            ) / 3
        }

        return fused_context

    def match_objects_to_entities(self, objects: List, entities: List) -> List[Dict]:
        """Match visual objects to linguistic entities"""
        matches = []

        for obj in objects:
            for entity in entities:
                # Simple matching based on name/content
                obj_name = obj.get('name', str(obj)) if isinstance(obj, dict) else str(obj)
                if entity.lower() in obj_name.lower():
                    matches.append({
                        'object': obj,
                        'entity': entity,
                        'match_confidence': 0.8  # Simplified confidence
                    })

        return matches

    def identify_target_objects(self, entities: List[str], objects: List) -> List:
        """Identify which objects are targets based on linguistic entities"""
        target_objects = []

        for entity in entities:
            for obj in objects:
                obj_name = obj.get('name', str(obj)) if isinstance(obj, dict) else str(obj)
                if entity.lower() in obj_name.lower():
                    target_objects.append(obj)

        return target_objects

    def recommend_action(self, command_interpretation: Dict, executed_actions: List) -> str:
        """Recommend the next action based on command interpretation"""
        intended_actions = command_interpretation.get('intended_action', [])
        target_objects = command_interpretation.get('target_objects', [])

        if 'grasp' in intended_actions or 'pick' in intended_actions:
            return 'grasp_object'
        elif 'move' in intended_actions or 'go' in intended_actions:
            return 'navigate_to_target'
        elif 'point' in intended_actions or 'show' in intended_actions:
            return 'point_to_object'
        elif target_objects:
            # If there are target objects but no clear action, approach them
            return 'approach_object'
        else:
            return 'wait_for_command'

    def compute_visual_attention(self, visual_features: Dict, linguistic_features: Dict) -> Dict:
        """Compute where visual attention should be focused"""
        attention_map = {
            'focus_objects': [],
            'spatial_attention': 'center',
            'attention_confidence': visual_features.get('confidence', 0.0)
        }

        # Focus on objects that match linguistic entities
        for entity in linguistic_features.get('entities', []):
            for obj in visual_features.get('objects', []):
                obj_name = obj.get('name', str(obj)) if isinstance(obj, dict) else str(obj)
                if entity.lower() in obj_name.lower():
                    attention_map['focus_objects'].append(obj)

        return attention_map

    def compute_linguistic_attention(self, linguistic_features: Dict, visual_features: Dict) -> Dict:
        """Compute where linguistic attention should be focused"""
        attention_map = {
            'focus_entities': linguistic_features.get('entities', []),
            'action_attention': linguistic_features.get('actions', []),
            'attention_confidence': linguistic_features.get('confidence', 0.0)
        }

        return attention_map

    def generate_fusion_output(self, fused_state: FusedRepresentation):
        """Generate appropriate output based on fused state"""
        if fused_state.overall_confidence < self.confidence_threshold:
            # Low confidence, ask for clarification
            output_msg = String()
            output_msg.data = f"CLARIFICATION_NEEDED: Confidence={fused_state.overall_confidence:.2f}"
            self.command_output_publisher.publish(output_msg)
            return

        # Determine the appropriate action based on fused context
        action_recommendation = fused_state.fused_context.get('action_recommendation', {}).get('recommended_action', 'wait')

        if action_recommendation == 'grasp_object':
            target_objects = fused_state.fused_context.get('command_interpretation', {}).get('target_objects', [])
            if target_objects:
                output_msg = String()
                output_msg.data = f"GRASP_OBJECT: {target_objects[0]}"
                self.command_output_publisher.publish(output_msg)

        elif action_recommendation == 'navigate_to_target':
            # Simplified navigation command
            output_msg = String()
            output_msg.data = "NAVIGATE: forward_1m"
            self.command_output_publisher.publish(output_msg)

        elif action_recommendation == 'point_to_object':
            target_objects = fused_state.fused_context.get('command_interpretation', {}).get('target_objects', [])
            if target_objects:
                output_msg = String()
                output_msg.data = f"POINT_TO: {target_objects[0]}"
                self.command_output_publisher.publish(output_msg)

        else:
            # Default to waiting for next command
            output_msg = String()
            output_msg.data = "WAIT: for_next_command"
            self.command_output_publisher.publish(output_msg)

        # Publish detailed fusion output
        fusion_output = {
            'timestamp': fused_state.timestamp,
            'visual_features': fused_state.visual_features,
            'linguistic_features': fused_state.linguistic_features,
            'action_features': fused_state.action_features,
            'fused_context': fused_state.fused_context,
            'overall_confidence': fused_state.overall_confidence
        }

        output_msg = String()
        output_msg.data = json.dumps(fusion_output)
        self.fused_output_publisher.publish(output_msg)

    def update_fusion_state(self):
        """Update the fusion state based on current conditions"""
        if not self.multimodal_inputs:
            self.fusion_state = FusionState.INACTIVE
        elif self.current_fused_state and self.current_fused_state.overall_confidence > self.confidence_threshold:
            self.fusion_state = FusionState.FUSION_ACTIVE
        else:
            # Determine state based on most recent input modality
            if self.multimodal_inputs:
                latest_input = self.multimodal_inputs[-1]
                if latest_input.modality == 'vision':
                    self.fusion_state = FusionState.VISION_PROCESSING
                elif latest_input.modality == 'language':
                    self.fusion_state = FusionState.LANGUAGE_PROCESSING
                elif latest_input.modality == 'action':
                    self.fusion_state = FusionState.EXECUTION

    def publish_attention_map(self):
        """Publish attention map showing focus areas"""
        if not self.current_fused_state:
            return

        attention_info = {
            'visual_attention': self.current_fused_state.fused_context.get('attention_focus', {}).get('visual_attention', {}),
            'linguistic_attention': self.current_fused_state.fused_context.get('attention_focus', {}).get('linguistic_attention', {}),
            'timestamp': time.time()
        }

        attention_msg = String()
        attention_msg.data = json.dumps(attention_info)
        self.attention_map_publisher.publish(attention_msg)

    def enable_multimodal_fusion(self, enable: bool):
        """Enable or disable multimodal fusion"""
        self.multimodal_fusion_enabled = enable
        self.get_logger().info(f"Multimodal fusion {'enabled' if enable else 'disabled'}")

    def set_attention_weight(self, connection: str, weight: float):
        """Set the attention weight for a cross-modal connection"""
        if connection in self.attention_weights:
            self.attention_weights[connection] = max(0.0, min(1.0, weight))
            self.get_logger().info(f"Attention weight {connection} set to {weight}")

    def get_fusion_stats(self) -> Dict[str, any]:
        """Get multimodal fusion statistics"""
        return {
            'fusion_state': self.fusion_state.value,
            'input_buffer_size': len(self.multimodal_inputs),
            'fused_representations_count': len(self.fused_representations),
            'overall_confidence': self.current_fused_state.overall_confidence if self.current_fused_state else 0.0,
            'enabled': self.multimodal_fusion_enabled,
            'attention_weights': self.attention_weights.copy()
        }

    def reset_fusion_system(self):
        """Reset the fusion system to initial state"""
        with self.data_lock:
            self.multimodal_inputs.clear()
            self.fused_representations.clear()
            self.current_fused_state = None
            self.fusion_state = FusionState.INACTIVE
            self.scene_context.clear()
            self.language_context.clear()
            self.action_context.clear()
            self.get_logger().info("Multimodal fusion system reset")


def main(args=None):
    rclpy.init(args=args)

    multimodal_fusion_node = MultimodalFusionNode()

    try:
        rclpy.spin(multimodal_fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        multimodal_fusion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()