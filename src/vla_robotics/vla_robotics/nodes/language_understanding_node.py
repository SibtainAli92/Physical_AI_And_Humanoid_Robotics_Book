#!/usr/bin/env python3

"""
Language Understanding Node for Humanoid Robot VLA System

This node processes natural language commands and converts them into
structured representations that can be used by the action planning system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Point
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
import re
from enum import Enum


class CommandCategory(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    INFORMATION = "information"
    SYSTEM = "system"


class CommandType(Enum):
    MOVE_TO = "move_to"
    GRASP = "grasp"
    RELEASE = "release"
    POINT = "point"
    SPEAK = "speak"
    FOLLOW = "follow"
    WAIT = "wait"
    STOP = "stop"
    GO = "go"
    COME = "come"


@dataclass
class ParsedCommand:
    """Represents a parsed natural language command"""
    id: str
    original_text: str
    category: CommandCategory
    type: CommandType
    entities: Dict[str, List[str]]
    spatial_relations: List[str]
    quantifiers: List[str]
    confidence: float
    timestamp: float
    parameters: Dict[str, any]


class LanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('language_understanding_node')

        # Publishers for language understanding output
        self.parsed_command_publisher = self.create_publisher(String, 'vla/parsed_command', 10)
        self.language_status_publisher = self.create_publisher(String, 'vla/language_status', 10)
        self.command_response_publisher = self.create_publisher(String, 'vla/command_response', 10)

        # Subscribers for language input
        self.speech_input_subscriber = self.create_subscription(
            String, 'speech_recognition/text', self.speech_input_callback, 10)
        self.text_command_subscriber = self.create_subscription(
            String, 'vla/text_command', self.text_command_callback, 10)

        # Timer for language processing
        self.language_timer = self.create_timer(0.1, self.language_processing_callback)  # 10 Hz

        # Language understanding state
        self.pending_commands = []
        self.processed_commands = []
        self.command_history = []
        self.language_understanding_enabled = True
        self.sim_time = time.time()
        self.last_language_time = time.time()

        # Language patterns and vocabularies
        self.navigation_patterns = [
            r'go to the (.+)',
            r'move to the (.+)',
            r'walk to the (.+)',
            r'navigate to the (.+)',
            r'go over to the (.+)',
            r'head to the (.+)',
            r'go (?:forward|backward|left|right)',
            r'move (?:forward|backward|left|right)',
            r'come here',
            r'come to me',
            r'follow me'
        ]

        self.manipulation_patterns = [
            r'(?:pick up|grasp|take|lift) the (.+)',
            r'(?:put down|place|release|drop) the (.+)',
            r'grab the (.+)',
            r'hold the (.+)',
            r'get the (.+)',
            r'bring me the (.+)'
        ]

        self.interaction_patterns = [
            r'point to the (.+)',
            r'point at the (.+)',
            r'show me the (.+)',
            r'look at the (.+)',
            r'wave to (.+)',
            r'greet (.+)',
            r'tell me about (.+)'
        ]

        self.information_patterns = [
            r'what is this',
            r'what do you see',
            r'tell me about your surroundings',
            r'describe the (.+)',
            r'how many (.+) are there',
            r'where is the (.+)'
        ]

        self.system_patterns = [
            r'stop',
            r'wait',
            r'pause',
            r'reset',
            r'help',
            r'what can you do'
        ]

        # Entity types
        self.object_entities = {
            'furniture': ['table', 'chair', 'sofa', 'bed', 'desk', 'couch'],
            'kitchen': ['cup', 'plate', 'bowl', 'bottle', 'glass', 'fork', 'spoon', 'knife'],
            'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white'],
            'sizes': ['small', 'big', 'large', 'tiny', 'huge', 'little', 'medium'],
            'spatial': ['left', 'right', 'front', 'back', 'behind', 'in front of', 'near', 'far', 'above', 'below']
        }

        # Spatial relation keywords
        self.spatial_relations = [
            'left', 'right', 'front', 'back', 'behind', 'in front of', 'near', 'far',
            'above', 'below', 'on', 'under', 'next to', 'beside', 'between', 'inside', 'outside'
        ]

        # Quantifier keywords
        self.quantifiers = [
            'one', 'two', 'three', 'several', 'many', 'few', 'all', 'some', 'any',
            'first', 'last', 'next', 'previous', 'other', 'another'
        ]

        # Command type mappings
        self.command_type_mappings = {
            'go': CommandType.MOVE_TO,
            'move': CommandType.MOVE_TO,
            'walk': CommandType.MOVE_TO,
            'navigate': CommandType.MOVE_TO,
            'head': CommandType.MOVE_TO,
            'pick': CommandType.GRASP,
            'grasp': CommandType.GRASP,
            'take': CommandType.GRASP,
            'lift': CommandType.GRASP,
            'grab': CommandType.GRASP,
            'put': CommandType.RELEASE,
            'place': CommandType.RELEASE,
            'release': CommandType.RELEASE,
            'drop': CommandType.RELEASE,
            'point': CommandType.POINT,
            'show': CommandType.POINT,
            'look': CommandType.POINT,
            'speak': CommandType.SPEAK,
            'say': CommandType.SPEAK,
            'tell': CommandType.SPEAK,
            'follow': CommandType.FOLLOW,
            'wait': CommandType.WAIT,
            'pause': CommandType.WAIT,
            'stop': CommandType.STOP
        }

        # Threading lock for data access
        self.data_lock = threading.Lock()

        self.get_logger().info('Language Understanding Node initialized')

    def speech_input_callback(self, msg):
        """Callback for speech recognition input"""
        with self.data_lock:
            if self.language_understanding_enabled:
                self.process_language_input(msg.data, source='speech')
                self.get_logger().info(f'Received speech command: {msg.data}')

    def text_command_callback(self, msg):
        """Callback for text command input"""
        with self.data_lock:
            if self.language_understanding_enabled:
                self.process_language_input(msg.data, source='text')
                self.get_logger().info(f'Received text command: {msg.data}')

    def language_processing_callback(self):
        """Main language processing callback"""
        if not self.language_understanding_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_language_time
        self.last_language_time = current_time

        with self.data_lock:
            # Process any pending commands
            while self.pending_commands:
                command = self.pending_commands.pop(0)
                self.processed_commands.append(command)

                # Publish parsed command
                parsed_msg = String()
                parsed_msg.data = f"PARSED: {command.type.value} - {command.original_text}"
                self.parsed_command_publisher.publish(parsed_msg)

                # Add to history
                self.command_history.append(command)
                if len(self.command_history) > 50:  # Keep last 50 commands
                    self.command_history.pop(0)

        # Publish language status
        status_msg = String()
        status_msg.data = f"LANGUAGE: Pending={len(self.pending_commands)}, Processed={len(self.processed_commands)}, History={len(self.command_history)}, Time={current_time:.2f}"
        self.language_status_publisher.publish(status_msg)

    def process_language_input(self, text: str, source: str = 'text'):
        """Process a natural language input and parse it into structured command"""
        if not text.strip():
            return

        # Preprocess the text
        clean_text = self.preprocess_text(text)

        # Parse the command
        parsed_command = self.parse_command(clean_text)

        if parsed_command:
            # Add to pending commands for processing
            self.pending_commands.append(parsed_command)

            # Log the parsed command
            self.get_logger().info(
                f'Parsed command: {parsed_command.type.value} '
                f'with entities: {parsed_command.entities} '
                f'and confidence: {parsed_command.confidence:.2f}'
            )

    def preprocess_text(self, text: str) -> str:
        """Preprocess text input for parsing"""
        # Convert to lowercase
        text = text.lower().strip()

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Expand contractions (simple expansions)
        contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "wouldn't": "would not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "mustn't": "must not",
            "needn't": "need not"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse a natural language command into structured representation"""
        # Determine command category
        category = self.categorize_command(text)

        # Determine command type
        command_type = self.identify_command_type(text)

        # Extract entities
        entities = self.extract_entities(text)

        # Extract spatial relations
        spatial_relations = self.extract_spatial_relations(text)

        # Extract quantifiers
        quantifiers = self.extract_quantifiers(text)

        # Calculate confidence based on pattern matches and entity extraction
        confidence = self.calculate_parsing_confidence(text, entities, spatial_relations)

        # Extract parameters based on command type
        parameters = self.extract_parameters(text, command_type, entities)

        # Create parsed command
        parsed_command = ParsedCommand(
            id=f"cmd_{int(time.time() * 1000)}",
            original_text=text,
            category=category,
            type=command_type,
            entities=entities,
            spatial_relations=spatial_relations,
            quantifiers=quantifiers,
            confidence=confidence,
            timestamp=time.time(),
            parameters=parameters
        )

        return parsed_command

    def categorize_command(self, text: str) -> CommandCategory:
        """Categorize the command into one of the main categories"""
        # Check navigation patterns
        for pattern in self.navigation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return CommandCategory.NAVIGATION

        # Check manipulation patterns
        for pattern in self.manipulation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return CommandCategory.MANIPULATION

        # Check interaction patterns
        for pattern in self.interaction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return CommandCategory.INTERACTION

        # Check information patterns
        for pattern in self.information_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return CommandCategory.INFORMATION

        # Check system patterns
        for pattern in self.system_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return CommandCategory.SYSTEM

        # Default to navigation if no specific category is identified
        return CommandCategory.NAVIGATION

    def identify_command_type(self, text: str) -> CommandType:
        """Identify the specific command type from text"""
        # Look for key verbs in the text
        words = text.split()
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.command_type_mappings:
                return self.command_type_mappings[clean_word]

        # Check for specific patterns
        if any(pattern in text for pattern in ['pick up', 'grasp', 'take', 'lift', 'grab']):
            return CommandType.GRASP
        elif any(pattern in text for pattern in ['put down', 'place', 'release', 'drop']):
            return CommandType.RELEASE
        elif any(pattern in text for pattern in ['point to', 'point at', 'show me', 'look at']):
            return CommandType.POINT
        elif any(pattern in text for pattern in ['follow me', 'come to me', 'come here']):
            return CommandType.FOLLOW
        elif any(pattern in text for pattern in ['stop', 'wait', 'pause']):
            return CommandType.WAIT
        elif any(pattern in text for pattern in ['speak', 'say', 'tell']):
            return CommandType.SPEAK
        else:
            # Default to move if it's a navigation-related command
            nav_keywords = ['go', 'move', 'walk', 'navigate', 'head', 'come']
            if any(keyword in text for keyword in nav_keywords):
                return CommandType.MOVE_TO

        return CommandType.WAIT  # Default fallback

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from the text"""
        entities = {
            'objects': [],
            'locations': [],
            'people': [],
            'colors': [],
            'sizes': [],
            'other': []
        }

        # Extract objects by checking against known entity types
        for entity_type, entity_list in self.object_entities.items():
            found_entities = []
            for entity in entity_list:
                if entity in text:
                    found_entities.append(entity)
            entities[entity_type if entity_type in entities else 'other'].extend(found_entities)

        # Extract colors specifically
        for color in self.object_entities['colors']:
            if color in text:
                entities['colors'].append(color)

        # Extract sizes specifically
        for size in self.object_entities['sizes']:
            if size in text:
                entities['sizes'].append(size)

        # Extract locations (simple approach)
        location_keywords = ['kitchen', 'living room', 'bedroom', 'bathroom', 'office', 'hallway', 'table', 'chair']
        for location in location_keywords:
            if location in text:
                entities['locations'].append(location)

        # Extract people (simple approach)
        people_keywords = ['me', 'you', 'person', 'man', 'woman', 'child', 'someone']
        for person in people_keywords:
            if person in text:
                entities['people'].append(person)

        return entities

    def extract_spatial_relations(self, text: str) -> List[str]:
        """Extract spatial relations from the text"""
        relations = []
        text_lower = text.lower()

        for relation in self.spatial_relations:
            if relation in text_lower:
                relations.append(relation)

        return relations

    def extract_quantifiers(self, text: str) -> List[str]:
        """Extract quantifiers from the text"""
        quantifiers = []
        text_lower = text.lower()

        for quantifier in self.quantifiers:
            if quantifier in text_lower:
                quantifiers.append(quantifier)

        return quantifiers

    def calculate_parsing_confidence(self, text: str, entities: Dict, spatial_relations: List[str]) -> float:
        """Calculate confidence in the parsing result"""
        confidence = 0.5  # Base confidence

        # Increase confidence if we found relevant entities
        entity_count = sum(len(v) for v in entities.values())
        if entity_count > 0:
            confidence += 0.2 * min(1.0, entity_count / 5.0)  # Up to 0.2 increase

        # Increase confidence if we found spatial relations
        if spatial_relations:
            confidence += 0.15 * min(1.0, len(spatial_relations) / 3.0)  # Up to 0.15 increase

        # Increase confidence if the text matches known patterns
        all_patterns = (self.navigation_patterns + self.manipulation_patterns +
                       self.interaction_patterns + self.information_patterns + self.system_patterns)
        pattern_matches = sum(1 for pattern in all_patterns if re.search(pattern, text, re.IGNORECASE))
        if pattern_matches > 0:
            confidence += 0.15  # Fixed increase for pattern match

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def extract_parameters(self, text: str, command_type: CommandType, entities: Dict[str, List[str]]) -> Dict[str, any]:
        """Extract specific parameters for the command type"""
        parameters = {}

        if command_type in [CommandType.MOVE_TO, CommandType.FOLLOW, CommandType.GO, CommandType.COME]:
            # Extract target location
            if entities['locations']:
                parameters['target_location'] = entities['locations'][0]
            elif entities['objects']:
                parameters['target_object'] = entities['objects'][0]

            # Extract spatial relations for navigation
            if self.spatial_relations:
                # This would be populated by the spatial relation extraction
                pass

        elif command_type in [CommandType.GRASP, CommandType.RELEASE]:
            # Extract target object
            if entities['objects']:
                parameters['target_object'] = entities['objects'][0]
            elif entities['colors'] or entities['sizes']:
                # Combine color and size information
                color = entities['colors'][0] if entities['colors'] else ''
                size = entities['sizes'][0] if entities['sizes'] else ''
                if color or size:
                    parameters['target_object'] = f"{size} {color}".strip()

        elif command_type == CommandType.POINT:
            # Extract target to point to
            if entities['objects']:
                parameters['target'] = entities['objects'][0]
            elif entities['locations']:
                parameters['target'] = entities['locations'][0]

        elif command_type == CommandType.SPEAK:
            # Extract text to speak
            # For now, we'll use the original text without command verbs
            words = text.split()
            # Remove command-related words
            content_words = [w for w in words if w not in ['say', 'speak', 'tell', 'please', 'can', 'you']]
            parameters['text'] = ' '.join(content_words)

        return parameters

    def generate_command_response(self, parsed_command: ParsedCommand) -> str:
        """Generate a natural language response to acknowledge the command"""
        if parsed_command.confidence < 0.5:
            return "I'm not sure I understood that command correctly."

        if parsed_command.type == CommandType.MOVE_TO:
            if 'target_object' in parsed_command.parameters:
                return f"Moving to the {parsed_command.parameters['target_object']}."
            elif 'target_location' in parsed_command.parameters:
                return f"Going to the {parsed_command.parameters['target_location']}."
            else:
                return "Moving to the specified location."

        elif parsed_command.type == CommandType.GRASP:
            target = parsed_command.parameters.get('target_object', 'object')
            return f"Attempting to grasp the {target}."

        elif parsed_command.type == CommandType.RELEASE:
            return "Releasing the object."

        elif parsed_command.type == CommandType.POINT:
            target = parsed_command.parameters.get('target', 'object')
            return f"Pointing to the {target}."

        elif parsed_command.type == CommandType.SPEAK:
            return f"I will say: {parsed_command.parameters.get('text', 'hello')}"

        elif parsed_command.type == CommandType.FOLLOW:
            return "I will follow you."

        elif parsed_command.type == CommandType.WAIT:
            return "Waiting for further instructions."

        elif parsed_command.type == CommandType.STOP:
            return "Stopping all actions."

        else:
            return f"Received command: {parsed_command.type.value}"

    def get_command_by_id(self, cmd_id: str) -> Optional[ParsedCommand]:
        """Get a specific command by its ID"""
        with self.data_lock:
            for cmd in self.command_history:
                if cmd.id == cmd_id:
                    return cmd
            return None

    def get_recent_commands(self, count: int = 5) -> List[ParsedCommand]:
        """Get the most recent commands"""
        with self.data_lock:
            return self.command_history[-count:]

    def enable_language_understanding(self, enable: bool):
        """Enable or disable language understanding"""
        self.language_understanding_enabled = enable
        self.get_logger().info(f"Language understanding {'enabled' if enable else 'disabled'}")

    def get_language_stats(self) -> Dict[str, any]:
        """Get language understanding statistics"""
        return {
            'pending_commands': len(self.pending_commands),
            'processed_commands': len(self.processed_commands),
            'command_history_size': len(self.command_history),
            'enabled': self.language_understanding_enabled,
            'last_command_time': self.last_language_time
        }


def main(args=None):
    rclpy.init(args=args)

    language_understanding_node = LanguageUnderstandingNode()

    try:
        rclpy.spin(language_understanding_node)
    except KeyboardInterrupt:
        pass
    finally:
        language_understanding_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()