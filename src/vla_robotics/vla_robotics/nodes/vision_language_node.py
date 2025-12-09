#!/usr/bin/env python3

"""
Vision-Language Node for Humanoid Robot VLA System

This node handles the integration of vision and language processing,
enabling the robot to understand visual scenes in the context of
natural language commands.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Point, Pose, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import time
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
import re


@dataclass
class VisualObject:
    """Represents a visually detected object with language associations"""
    id: str
    name: str
    position: Point
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    color: str
    size: str  # small, medium, large
    spatial_relation: str  # left, right, front, back, near, far


class VisionLanguageNode(Node):
    def __init__(self):
        super().__init__('vision_language_node')

        # Publishers for vision-language integration
        self.vision_language_status_publisher = self.create_publisher(String, 'vla/status', 10)
        self.scene_description_publisher = self.create_publisher(String, 'vla/scene_description', 10)
        self.object_grounding_publisher = self.create_publisher(MarkerArray, 'vla/object_grounding', 10)
        self.language_response_publisher = self.create_publisher(String, 'vla/language_response', 10)

        # Subscribers for vision and language inputs
        self.camera_subscriber = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.language_command_subscriber = self.create_subscription(
            String, 'vla/language_command', self.language_command_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Timer for vision-language processing
        self.vl_timer = self.create_timer(0.2, self.vision_language_callback)  # 5 Hz

        # Vision-language state
        self.cv_bridge = CvBridge()
        self.current_image = None
        self.current_language_command = ""
        self.current_language_timestamp = 0.0
        self.visual_objects = []
        self.language_entities = []
        self.vision_language_enabled = True
        self.sim_time = time.time()
        self.last_vl_time = time.time()

        # Vision processing parameters
        self.object_detection_threshold = 0.7
        self.color_names = {
            (255, 0, 0): "red",
            (0, 255, 0): "green",
            (0, 0, 255): "blue",
            (255, 255, 0): "yellow",
            (255, 0, 255): "magenta",
            (0, 255, 255): "cyan",
            (128, 128, 128): "gray",
            (192, 192, 192): "white",
            (0, 0, 0): "black"
        }

        # Language processing parameters
        self.spatial_keywords = {
            'left', 'right', 'front', 'back', 'behind', 'in front of', 'near', 'far',
            'above', 'below', 'on', 'under', 'next to', 'beside'
        }
        self.size_keywords = {
            'small', 'tiny', 'little', 'big', 'large', 'huge', 'giant',
            'medium', 'average', 'normal'
        }
        self.color_keywords = {
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink',
            'brown', 'black', 'white', 'gray', 'grey'
        }

        # Threading lock for data access
        self.data_lock = threading.Lock()

        self.get_logger().info('Vision-Language Node initialized')

    def camera_callback(self, msg):
        """Callback for camera image data"""
        with self.data_lock:
            try:
                # Convert ROS Image to OpenCV format
                self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'Error converting image: {e}')

    def language_command_callback(self, msg):
        """Callback for language commands"""
        with self.data_lock:
            self.current_language_command = msg.data
            self.current_language_timestamp = time.time()
            self.get_logger().info(f'Received language command: {msg.data}')

    def laser_callback(self, msg):
        """Callback for laser scan data (for spatial reasoning)"""
        with self.data_lock:
            self.current_laser_scan = msg

    def vision_language_callback(self):
        """Main vision-language integration callback"""
        if not self.vision_language_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_vl_time
        self.last_vl_time = current_time

        with self.data_lock:
            # Process vision data if available
            if self.current_image is not None:
                self.visual_objects = self.process_image_for_objects(self.current_image)

            # Process language command if available
            if self.current_language_command:
                self.language_entities = self.extract_language_entities(self.current_language_command)

            # Integrate vision and language
            if self.visual_objects and self.language_entities:
                self.integrate_vision_language()

        # Publish scene description
        self.publish_scene_description()

        # Publish object grounding visualization
        self.publish_object_grounding()

        # Publish vision-language status
        status_msg = String()
        status_msg.data = f"VLA: Objects={len(self.visual_objects)}, Language_Entities={len(self.language_entities)}, Time={current_time:.2f}"
        self.vision_language_status_publisher.publish(status_msg)

    def process_image_for_objects(self, image):
        """Process image to detect and describe objects"""
        detected_objects = []

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect colored objects using color thresholds (simplified approach)
        # In a real system, this would use a trained object detection model
        for color_bgr, color_name in self.color_names.items():
            # Convert BGR to HSV for thresholding
            bgr_array = np.uint8([[color_bgr]])
            hsv_color = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]

            # Create a range around this color
            lower = np.array([max(0, hsv_color[0]-10), 50, 50])
            upper = np.array([min(180, hsv_color[0]+10), 255, 255])

            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Filter small noise
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Determine size category
                    size_category = self.categorize_size(w * h)

                    # Determine spatial relation based on position in image
                    spatial_relation = self.determine_spatial_relation(center_x, center_y, image.shape[1], image.shape[0])

                    # Calculate confidence based on size
                    confidence = min(1.0, area / 5000.0)

                    obj = VisualObject(
                        id=f"{color_name}_obj_{len(detected_objects)}",
                        name=f"{color_name} object",
                        position=Point(x=float(center_x), y=float(center_y), z=0.0),
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        color=color_name,
                        size=size_category,
                        spatial_relation=spatial_relation
                    )
                    detected_objects.append(obj)

        # Also detect basic shapes (simplified)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Minimum area threshold
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                # Determine shape type based on number of vertices
                num_vertices = len(approx)
                if num_vertices == 3:
                    shape_name = "triangle"
                elif num_vertices == 4:
                    # Check if it's more like a square or rectangle
                    aspect_ratio = float(w) / h
                    shape_name = "square" if 0.75 <= aspect_ratio <= 1.25 else "rectangle"
                elif num_vertices > 4:
                    shape_name = "circle" if abs(1 - (w/h)) < 0.2 else "ellipse"
                else:
                    shape_name = "object"

                # Calculate confidence based on how "shape-like" it is
                confidence = min(1.0, area / 3000.0)

                # Determine spatial relation
                spatial_relation = self.determine_spatial_relation(center_x, center_y, image.shape[1], image.shape[0])

                obj = VisualObject(
                    id=f"{shape_name}_obj_{len(detected_objects)}",
                    name=shape_name,
                    position=Point(x=float(center_x), y=float(center_y), z=0.0),
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    color="unknown",
                    size=self.categorize_size(w * h),
                    spatial_relation=spatial_relation
                )
                detected_objects.append(obj)

        return detected_objects

    def categorize_size(self, pixel_area):
        """Categorize object size based on pixel area"""
        if pixel_area < 500:
            return "small"
        elif pixel_area < 2000:
            return "medium"
        else:
            return "large"

    def determine_spatial_relation(self, x, y, img_width, img_height):
        """Determine spatial relation based on position in image"""
        # Normalize coordinates
        norm_x = x / img_width
        norm_y = y / img_height

        # Determine horizontal relation
        if norm_x < 0.33:
            horizontal = "left"
        elif norm_x > 0.67:
            horizontal = "right"
        else:
            horizontal = "center"

        # Determine vertical relation
        if norm_y < 0.33:
            vertical = "top"
        elif norm_y > 0.67:
            vertical = "bottom"
        else:
            vertical = "middle"

        # Combine for full spatial relation
        if horizontal == "center" and vertical == "middle":
            return "center"
        elif horizontal == "center":
            return vertical
        elif vertical == "middle":
            return horizontal
        else:
            return f"{vertical} {horizontal}"

    def extract_language_entities(self, command):
        """Extract entities and spatial relations from language command"""
        entities = []

        # Simple entity extraction using keyword matching
        # In a real system, this would use NLP models
        command_lower = command.lower()

        # Extract spatial keywords
        for keyword in self.spatial_keywords:
            if keyword in command_lower:
                entities.append(('spatial_relation', keyword))

        # Extract size keywords
        for keyword in self.size_keywords:
            if keyword in command_lower:
                entities.append(('size', keyword))

        # Extract color keywords
        for keyword in self.color_keywords:
            if keyword in command_lower:
                entities.append(('color', keyword))

        # Extract potential object names (simple approach)
        # Look for common object words
        object_keywords = ['object', 'thing', 'item', 'box', 'ball', 'cup', 'table', 'chair', 'person']
        for keyword in object_keywords:
            if keyword in command_lower:
                entities.append(('object', keyword))

        # Extract numbers
        numbers = re.findall(r'\d+', command)
        for num in numbers:
            entities.append(('number', num))

        return entities

    def integrate_vision_language(self):
        """Integrate vision and language information"""
        # Match language entities to visual objects
        for entity_type, entity_value in self.language_entities:
            if entity_type == 'color':
                # Find objects matching the color
                matching_objects = [obj for obj in self.visual_objects if obj.color == entity_value]
                if matching_objects:
                    self.get_logger().info(f"Found {len(matching_objects)} {entity_value} objects")

            elif entity_type == 'size':
                # Find objects matching the size
                matching_objects = [obj for obj in self.visual_objects if obj.size == entity_value]
                if matching_objects:
                    self.get_logger().info(f"Found {len(matching_objects)} {entity_value} objects")

            elif entity_type == 'spatial_relation':
                # Find objects matching the spatial relation
                matching_objects = [obj for obj in self.visual_objects if entity_value in obj.spatial_relation]
                if matching_objects:
                    self.get_logger().info(f"Found {len(matching_objects)} objects in {entity_value} relation")

    def publish_scene_description(self):
        """Publish a textual description of the current scene"""
        if not self.visual_objects:
            return

        # Create a simple scene description
        color_counts = {}
        size_counts = {}
        position_groups = {}

        for obj in self.visual_objects:
            # Count colors
            if obj.color not in color_counts:
                color_counts[obj.color] = 0
            color_counts[obj.color] += 1

            # Count sizes
            if obj.size not in size_counts:
                size_counts[obj.size] = 0
            size_counts[obj.size] += 1

            # Group by position
            if obj.spatial_relation not in position_groups:
                position_groups[obj.spatial_relation] = []
            position_groups[obj.spatial_relation].append(obj.name)

        # Create description
        description_parts = []
        if color_counts:
            color_desc = ", ".join([f"{count} {color}" for color, count in color_counts.items() if color != "unknown"])
            if color_desc:
                description_parts.append(f"Colors: {color_desc}")

        if size_counts:
            size_desc = ", ".join([f"{count} {size}" for size, count in size_counts.items()])
            if size_desc:
                description_parts.append(f"Sizes: {size_desc}")

        if position_groups:
            pos_desc = []
            for pos, objects in position_groups.items():
                if objects:
                    obj_str = ", ".join(objects[:3])  # Limit to first 3 objects
                    if len(objects) > 3:
                        obj_str += f" and {len(objects)-3} more"
                    pos_desc.append(f"{pos}: {obj_str}")
            if pos_desc:
                description_parts.append(f"Positions: {'; '.join(pos_desc)}")

        if description_parts:
            description = "Scene contains: " + "; ".join(description_parts)
        else:
            description = "Scene: No distinctive objects detected"

        desc_msg = String()
        desc_msg.data = description
        self.scene_description_publisher.publish(desc_msg)

    def publish_object_grounding(self):
        """Publish visualization markers for object grounding"""
        marker_array = MarkerArray()
        marker_id = 0

        for obj in self.visual_objects:
            # Create a marker for the object
            marker = Marker()
            marker.header.frame_id = "camera_frame"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position (convert image coordinates to a relative position)
            # For visualization, we'll use a relative scale
            marker.pose.position.x = (obj.position.x - 320) / 1000  # Assuming 640x480 image, scale appropriately
            marker.pose.position.y = (obj.position.y - 240) / 1000
            marker.pose.position.z = 1.0  # Fixed distance for visualization

            # Scale based on bounding box size
            marker.scale.x = max(0.05, obj.bbox[2] / 1000)
            marker.scale.y = max(0.05, obj.bbox[3] / 1000)
            marker.scale.z = 0.1

            # Color based on object color
            color_rgb = self.get_color_rgb(obj.color)
            marker.color.r = color_rgb[0] / 255.0
            marker.color.g = color_rgb[1] / 255.0
            marker.color.b = color_rgb[2] / 255.0
            marker.color.a = min(1.0, obj.confidence + 0.3)  # Ensure some visibility

            marker_array.markers.append(marker)

            # Add text label
            text_marker = Marker()
            text_marker.header.frame_id = "camera_frame"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = marker.pose.position.z + 0.1  # Above the object
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.1  # Text scale
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"{obj.name}\n{obj.confidence:.2f}"

            marker_array.markers.append(text_marker)

        self.object_grounding_publisher.publish(marker_array)

    def get_color_rgb(self, color_name):
        """Get RGB values for a color name"""
        color_map = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'cyan': (0, 255, 255),
            'gray': (128, 128, 128),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'unknown': (192, 192, 192)
        }
        return color_map.get(color_name, (192, 192, 192))

    def process_language_command(self, command):
        """Process a language command and return relevant visual objects"""
        with self.data_lock:
            # Extract entities from the command
            entities = self.extract_language_entities(command)

            # Find matching objects based on entities
            matching_objects = []

            for entity_type, entity_value in entities:
                if entity_type == 'color':
                    matching_objects.extend([obj for obj in self.visual_objects if obj.color == entity_value])
                elif entity_type == 'size':
                    matching_objects.extend([obj for obj in self.visual_objects if obj.size == entity_value])
                elif entity_type == 'spatial_relation':
                    matching_objects.extend([obj for obj in self.visual_objects if entity_value in obj.spatial_relation])

            # Remove duplicates
            unique_objects = []
            seen_ids = set()
            for obj in matching_objects:
                if obj.id not in seen_ids:
                    unique_objects.append(obj)
                    seen_ids.add(obj.id)

            return unique_objects

    def enable_vision_language(self, enable: bool):
        """Enable or disable vision-language processing"""
        self.vision_language_enabled = enable
        self.get_logger().info(f"Vision-language processing {'enabled' if enable else 'disabled'}")

    def get_vision_language_stats(self) -> Dict[str, any]:
        """Get vision-language system statistics"""
        return {
            'object_count': len(self.visual_objects),
            'entity_count': len(self.language_entities),
            'last_command': self.current_language_command,
            'enabled': self.vision_language_enabled
        }


def main(args=None):
    rclpy.init(args=args)

    vision_language_node = VisionLanguageNode()

    try:
        rclpy.spin(vision_language_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_language_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()