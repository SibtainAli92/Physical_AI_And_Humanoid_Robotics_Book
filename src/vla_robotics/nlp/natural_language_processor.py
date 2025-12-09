#!/usr/bin/env python3
# natural_language_processor.py
# Natural language processing for humanoid robot

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from humanoid_msgs.msg import HumanoidControlCommand  # Custom message type
import speech_recognition as sr
import pyttsx3
import nltk
import spacy
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
import re
from dataclasses import dataclass
import json


@dataclass
class NLUResult:
    """Data class for Natural Language Understanding results"""
    intent: str
    entities: Dict[str, str]
    confidence: float
    action: Optional[str] = None
    parameters: Optional[Dict] = None


class NaturalLanguageProcessor(Node):
    """
    Natural language processing system for humanoid robot
    """
    def __init__(self):
        super().__init__('natural_language_processor')

        # Declare parameters
        self.declare_parameter('stt_language', 'en-US')
        self.declare_parameter('tts_language', 'en')
        self.declare_parameter('nlu_confidence_threshold', 0.7)
        self.declare_parameter('enable_context_awareness', True)

        self.stt_language = self.get_parameter('stt_language').value
        self.tts_language = self.get_parameter('tts_language').value
        self.nlu_confidence_threshold = self.get_parameter('nlu_confidence_threshold').value
        self.enable_context_awareness = self.get_parameter('enable_context_awareness').value

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level

        # Initialize NLP model (using spaCy as example)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.get_logger().warn("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Publishers
        self.speech_pub = self.create_publisher(
            String,
            '/nlp/speech_recognized',
            QoSProfile(depth=10)
        )

        self.response_pub = self.create_publisher(
            String,
            '/nlp/response',
            QoSProfile(depth=10)
        )

        self.command_pub = self.create_publisher(
            HumanoidControlCommand,
            '/nlp/robot_command',
            QoSProfile(depth=10)
        )

        # Subscribers
        self.text_command_sub = self.create_subscription(
            String,
            '/nlp/text_command',
            self.text_command_callback,
            QoSProfile(depth=10)
        )

        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio/input',
            self.audio_callback,
            QoSProfile(depth=10)
        )

        # Initialize internal state
        self.context = {}
        self.command_history = []
        self.processing_lock = threading.Lock()

        # Initialize intent patterns
        self.initialize_intent_patterns()

        self.get_logger().info('Natural Language Processor initialized')

    def initialize_intent_patterns(self):
        """
        Initialize patterns for intent recognition
        """
        self.intent_patterns = {
            'greeting': [
                r'hello|hi|hey|good morning|good afternoon|good evening',
                r'how are you|how do you do|what\'s up'
            ],
            'movement': [
                r'go to|move to|walk to|navigate to',
                r'forward|backward|left|right|turn',
                r'come here|approach|move closer'
            ],
            'action': [
                r'pick up|grasp|take|hold|grab',
                r'put down|release|drop|let go',
                r'wave|nod|shake|bow|greet',
                r'sit down|stand up|crouch|jump'
            ],
            'question': [
                r'what|where|when|who|why|how',
                r'can you|could you|would you',
                r'tell me about|explain|describe'
            ],
            'stop': [
                r'stop|halt|pause|freeze|cease|quit|exit'
            ]
        }

    def audio_callback(self, msg: AudioData):
        """
        Callback for audio input
        """
        # In a real implementation, this would process audio data
        # For now, we'll simulate speech recognition
        with self.processing_lock:
            try:
                # Convert audio data to text (simulated)
                recognized_text = self.simulate_speech_recognition(msg)

                if recognized_text:
                    self.process_text(recognized_text)
            except Exception as e:
                self.get_logger().error(f'Error processing audio: {str(e)}')

    def text_command_callback(self, msg: String):
        """
        Callback for text commands
        """
        with self.processing_lock:
            self.process_text(msg.data)

    def simulate_speech_recognition(self, audio_msg) -> Optional[str]:
        """
        Simulate speech recognition from audio data
        In a real implementation, this would use actual speech recognition
        """
        # This is a simulation - in reality, you'd process the actual audio data
        # For demo purposes, we'll return a fixed response
        return "Hello robot, please move forward"

    def process_text(self, text: str):
        """
        Process input text through the NLP pipeline
        """
        self.get_logger().info(f'Processing text: {text}')

        # Publish the recognized speech
        speech_msg = String()
        speech_msg.data = text
        self.speech_pub.publish(speech_msg)

        # Perform natural language understanding
        nlu_result = self.perform_nlu(text)

        if nlu_result and nlu_result.confidence >= self.nlu_confidence_threshold:
            # Generate response
            response = self.generate_response(nlu_result, text)

            # Publish response
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

            # Execute command if applicable
            self.execute_command(nlu_result)

            # Update context
            if self.enable_context_awareness:
                self.update_context(nlu_result, text)
        else:
            # Low confidence response
            response_msg = String()
            response_msg.data = "I didn't understand that. Could you please repeat?"
            self.response_pub.publish(response_msg)

    def perform_nlu(self, text: str) -> Optional[NLUResult]:
        """
        Perform Natural Language Understanding on input text
        """
        if self.nlp:
            # Use spaCy for NLP processing
            doc = self.nlp(text)

            # Extract entities
            entities = {}
            for ent in doc.ents:
                entities[ent.label_] = ent.text

            # Identify intent based on patterns
            intent = self.identify_intent(text)

            # Calculate confidence (simplified)
            confidence = 0.8 if intent != 'unknown' else 0.3

            # Extract action and parameters
            action = self.extract_action(text)
            parameters = self.extract_parameters(text)

            return NLUResult(
                intent=intent,
                entities=entities,
                confidence=confidence,
                action=action,
                parameters=parameters
            )
        else:
            # Fallback simple pattern matching
            intent = self.identify_intent(text)
            entities = self.extract_entities_simple(text)
            confidence = 0.7 if intent != 'unknown' else 0.2

            return NLUResult(
                intent=intent,
                entities=entities,
                confidence=confidence
            )

    def identify_intent(self, text: str) -> str:
        """
        Identify the intent of the input text using pattern matching
        """
        text_lower = text.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent

        return 'unknown'

    def extract_entities_simple(self, text: str) -> Dict[str, str]:
        """
        Simple entity extraction using regex patterns
        """
        entities = {}

        # Extract locations
        location_patterns = [
            r'to (\w+)',  # "go to kitchen"
            r'(\w+) room',  # "kitchen room"
            r'near (\w+)',  # "near the table"
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['LOCATION'] = match.group(1)

        # Extract objects
        object_patterns = [
            r'(\w+) object',  # "red object"
            r'pick up (\w+)',  # "pick up the cup"
            r'(\w+) cup',  # "blue cup"
        ]

        for pattern in object_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['OBJECT'] = match.group(1)

        return entities

    def extract_action(self, text: str) -> Optional[str]:
        """
        Extract action from text
        """
        actions = ['move', 'walk', 'turn', 'grasp', 'wave', 'speak', 'stop', 'sit', 'stand']
        text_lower = text.lower()

        for action in actions:
            if action in text_lower:
                return action

        return None

    def extract_parameters(self, text: str) -> Optional[Dict]:
        """
        Extract parameters from text
        """
        params = {}

        # Extract direction
        if 'forward' in text.lower():
            params['direction'] = 'forward'
        elif 'backward' in text.lower():
            params['direction'] = 'backward'
        elif 'left' in text.lower():
            params['direction'] = 'left'
        elif 'right' in text.lower():
            params['direction'] = 'right'

        # Extract distance if specified
        distance_match = re.search(r'(\d+(?:\.\d+)?) (meters|meter|m)', text, re.IGNORECASE)
        if distance_match:
            params['distance'] = float(distance_match.group(1))

        # Extract angle if specified
        angle_match = re.search(r'(\d+(?:\.\d+)?) (degrees|degree|deg)', text, re.IGNORECASE)
        if angle_match:
            params['angle'] = float(angle_match.group(1))

        return params if params else None

    def generate_response(self, nlu_result: NLUResult, original_text: str) -> str:
        """
        Generate a response based on NLU result
        """
        intent = nlu_result.intent

        if intent == 'greeting':
            return "Hello! How can I assist you today?"
        elif intent == 'movement':
            if nlu_result.parameters and 'direction' in nlu_result.parameters:
                direction = nlu_result.parameters['direction']
                return f"Moving {direction} as requested."
            else:
                return "I can move in various directions. Please specify where you'd like me to go."
        elif intent == 'action':
            if nlu_result.action:
                return f"Performing {nlu_result.action} action."
            else:
                return "I can perform various actions. What would you like me to do?"
        elif intent == 'question':
            return "I'm processing your question. Could you please be more specific?"
        elif intent == 'stop':
            return "Stopping all actions."
        else:
            return "I understand. How else may I help you?"

    def execute_command(self, nlu_result: NLUResult):
        """
        Execute robot command based on NLU result
        """
        if nlu_result.intent == 'movement' and nlu_result.parameters:
            # Create and publish a humanoid control command
            cmd_msg = HumanoidControlCommand()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.header.frame_id = 'base_link'

            # Set control mode based on parameters
            if 'direction' in nlu_result.parameters:
                direction = nlu_result.parameters['direction']

                # Map directions to velocity commands
                if direction == 'forward':
                    cmd_msg.joint_velocities = [0.1] * 24  # Simplified
                elif direction == 'backward':
                    cmd_msg.joint_velocities = [-0.1] * 24
                elif direction == 'left':
                    cmd_msg.joint_velocities = [0.05] * 24
                elif direction == 'right':
                    cmd_msg.joint_velocities = [-0.05] * 24

            cmd_msg.control_mode = 'velocity'
            self.command_pub.publish(cmd_msg)

    def update_context(self, nlu_result: NLUResult, text: str):
        """
        Update conversation context
        """
        self.context['last_intent'] = nlu_result.intent
        self.context['last_text'] = text
        self.context['timestamp'] = self.get_clock().now().nanoseconds


class SpeechToText(Node):
    """
    Speech-to-text component
    """
    def __init__(self):
        super().__init__('speech_to_text')

        # Publishers
        self.text_pub = self.create_publisher(
            String,
            '/stt/text_output',
            QoSProfile(depth=10)
        )

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Start listening thread
        self.listening = True
        self.listen_thread = threading.Thread(target=self.listen_continuously)
        self.listen_thread.start()

        self.get_logger().info('Speech-to-Text initialized')

    def listen_continuously(self):
        """
        Continuously listen for speech input
        """
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=5.0)

                # Recognize speech
                text = self.recognizer.recognize_google(audio)

                # Publish recognized text
                text_msg = String()
                text_msg.data = text
                self.text_pub.publish(text_msg)

                self.get_logger().info(f'Recognized: {text}')

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except sr.UnknownValueError:
                self.get_logger().info('Could not understand audio')
            except sr.RequestError as e:
                self.get_logger().error(f'Speech recognition error: {e}')
            except Exception as e:
                self.get_logger().error(f'Error in speech recognition: {e}')

    def destroy_node(self):
        """
        Clean up resources
        """
        self.listening = False
        if self.listen_thread.is_alive():
            self.listen_thread.join()
        super().destroy_node()


class TextToSpeech(Node):
    """
    Text-to-speech component
    """
    def __init__(self):
        super().__init__('text_to_speech')

        # Subscribers
        self.text_sub = self.create_subscription(
            String,
            '/tts/text_input',
            self.text_callback,
            QoSProfile(depth=10)
        )

        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()

        # Get and set available voices
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Set a female voice if available, otherwise use the first
            for voice in voices:
                if 'en' in voice.languages:
                    self.tts_engine.setProperty('voice', voice.id)
                    break

        self.tts_lock = threading.Lock()

        self.get_logger().info('Text-to-Speech initialized')

    def text_callback(self, msg: String):
        """
        Callback for text input to be spoken
        """
        with self.tts_lock:
            try:
                self.get_logger().info(f'Speaking: {msg.data}')
                self.tts_engine.say(msg.data)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.get_logger().error(f'Error in text-to-speech: {e}')


def main(args=None):
    rclpy.init(args=args)

    # Create NLP nodes
    nlp_processor = NaturalLanguageProcessor()
    stt = SpeechToText()
    tts = TextToSpeech()

    # Use a MultiThreadedExecutor to handle callbacks from multiple nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(nlp_processor)
    executor.add_node(stt)
    executor.add_node(tts)

    try:
        executor.spin()
    except KeyboardInterrupt:
        nlp_processor.get_logger().info('NLP processor interrupted by user')
        stt.get_logger().info('Speech-to-text interrupted by user')
        tts.get_logger().info('Text-to-speech interrupted by user')
    finally:
        stt.destroy_node()  # This will stop the listening thread
        nlp_processor.destroy_node()
        tts.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()