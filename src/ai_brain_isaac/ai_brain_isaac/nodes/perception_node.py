#!/usr/bin/env python3

"""
Perception Node for Humanoid Robot AI Brain

This node handles computer vision, object detection, and sensor data processing
using AI techniques similar to those in NVIDIA Isaac.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan, Imu, JointState
from geometry_msgs.msg import Point, Vector3, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import time
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Represents a detected object in the environment"""
    id: str
    type: str
    position: Point
    confidence: float
    size: Vector3
    timestamp: float


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Publishers for perception data
        self.object_detection_publisher = self.create_publisher(PoseArray, 'ai/detected_objects', 10)
        self.perception_status_publisher = self.create_publisher(String, 'ai/perception_status', 10)
        self.visualization_publisher = self.create_publisher(MarkerArray, 'ai/perception_viz', 10)
        self.depth_image_publisher = self.create_publisher(Image, 'ai/processed_depth', 10)

        # Subscribers for sensor data
        self.camera_subscriber = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.depth_subscriber = self.create_subscription(
            Image, 'camera/depth_image_raw', self.depth_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.pointcloud_subscriber = self.create_subscription(
            PointCloud2, 'point_cloud', self.pointcloud_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Timer for perception processing
        self.perception_timer = self.create_timer(0.1, self.perception_callback)  # 10 Hz

        # Perception state
        self.cv_bridge = CvBridge()
        self.current_image = None
        self.current_depth = None
        self.current_laser_scan = None
        self.current_pointcloud = None
        self.current_imu = None
        self.detected_objects = []
        self.perception_enabled = True
        self.sim_time = time.time()
        self.last_perception_time = time.time()

        # AI perception parameters
        self.object_detection_threshold = 0.7
        self.tracking_enabled = True
        self.segmentation_enabled = False
        self.feature_extraction_enabled = True

        # Object tracking
        self.tracked_objects = {}
        self.next_object_id = 0

        # Threading lock for data access
        self.data_lock = threading.Lock()

        self.get_logger().info('Perception Node initialized')

    def camera_callback(self, msg):
        """Callback for camera image data"""
        with self.data_lock:
            try:
                # Convert ROS Image to OpenCV format
                self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'Error converting image: {e}')

    def depth_callback(self, msg):
        """Callback for depth image data"""
        with self.data_lock:
            try:
                # Convert ROS Image to OpenCV format
                self.current_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            except Exception as e:
                self.get_logger().error(f'Error converting depth image: {e}')

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        with self.data_lock:
            self.current_laser_scan = msg

    def pointcloud_callback(self, msg):
        """Callback for point cloud data"""
        with self.data_lock:
            self.current_pointcloud = msg

    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.data_lock:
            self.current_imu = msg

    def perception_callback(self):
        """Main perception processing callback"""
        if not self.perception_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_perception_time
        self.last_perception_time = current_time

        detected_objects = []

        with self.data_lock:
            # Process camera image for object detection
            if self.current_image is not None:
                camera_objects = self.process_camera_image(self.current_image)
                detected_objects.extend(camera_objects)

            # Process laser scan for object detection
            if self.current_laser_scan is not None:
                laser_objects = self.process_laser_scan(self.current_laser_scan)
                detected_objects.extend(laser_objects)

            # Process point cloud if available
            if self.current_pointcloud is not None:
                pc_objects = self.process_pointcloud(self.current_pointcloud)
                detected_objects.extend(pc_objects)

        # Update tracked objects
        self.update_tracked_objects(detected_objects)

        # Publish detected objects
        self.publish_detected_objects()

        # Publish visualization
        self.publish_visualization()

        # Publish perception status
        status_msg = String()
        status_msg.data = f"PERCEPTION: Objects={len(self.tracked_objects)}, Time={current_time:.2f}"
        self.perception_status_publisher.publish(status_msg)

    def process_camera_image(self, image):
        """Process camera image for object detection and feature extraction"""
        detected_objects = []

        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple object detection using color thresholds (simplified approach)
        # In a real system, this would use a trained neural network
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect red objects (simplified)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # Find contours of red objects
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                # Calculate confidence based on size (larger objects are more likely to be real)
                confidence = min(1.0, area / 10000.0)

                if confidence > self.object_detection_threshold:
                    # Create detected object (position is relative to image center)
                    obj = DetectedObject(
                        id=f"red_obj_{len(detected_objects)}",
                        type="red_object",
                        position=Point(x=center_x, y=center_y, z=0.0),
                        confidence=confidence,
                        size=Vector3(x=float(w), y=float(h), z=0.1),  # z is depth (simplified)
                        timestamp=time.time()
                    )
                    detected_objects.append(obj)

        # Detect other colors similarly (simplified)
        # Blue objects
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                confidence = min(1.0, area / 10000.0)

                if confidence > self.object_detection_threshold:
                    obj = DetectedObject(
                        id=f"blue_obj_{len(detected_objects)}",
                        type="blue_object",
                        position=Point(x=center_x, y=center_y, z=0.0),
                        confidence=confidence,
                        size=Vector3(x=float(w), y=float(h), z=0.1),
                        timestamp=time.time()
                    )
                    detected_objects.append(obj)

        return detected_objects

    def process_laser_scan(self, scan_msg):
        """Process laser scan data for object detection"""
        detected_objects = []
        ranges = scan_msg.ranges

        # Simple clustering to detect objects in laser scan
        clusters = []
        current_cluster = []

        for i, range_val in enumerate(ranges):
            if not math.isnan(range_val) and range_val < scan_msg.range_max * 0.8:  # Valid range
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                if current_cluster and self.distance_to_cluster((x, y), current_cluster) < 0.3:  # 30cm threshold
                    current_cluster.append((x, y))
                else:
                    if len(current_cluster) > 2:  # At least 3 points to form an object
                        clusters.append(current_cluster)
                    current_cluster = [(x, y)]

        if len(current_cluster) > 2:
            clusters.append(current_cluster)

        # Process each cluster as a potential object
        for cluster in clusters:
            if len(cluster) > 2:
                # Calculate centroid of cluster
                avg_x = sum(p[0] for p in cluster) / len(cluster)
                avg_y = sum(p[1] for p in cluster) / len(cluster)

                # Calculate approximate size
                min_x = min(p[0] for p in cluster)
                max_x = max(p[0] for p in cluster)
                min_y = min(p[1] for p in cluster)
                max_y = max(p[1] for p in cluster)

                size_x = max_x - min_x
                size_y = max_y - min_y

                # Calculate confidence based on cluster size
                confidence = min(1.0, len(cluster) / 50.0)  # Higher confidence for larger clusters

                if confidence > self.object_detection_threshold * 0.5:  # Lower threshold for laser
                    obj = DetectedObject(
                        id=f"laser_obj_{len(detected_objects)}",
                        type="obstacle",
                        position=Point(x=avg_x, y=avg_y, z=0.0),
                        confidence=confidence,
                        size=Vector3(x=size_x, y=size_y, z=1.0),  # z is height (simplified)
                        timestamp=time.time()
                    )
                    detected_objects.append(obj)

        return detected_objects

    def process_pointcloud(self, pc_msg):
        """Process point cloud data for object detection"""
        # Simplified point cloud processing
        # In a real system, this would use more sophisticated algorithms
        detected_objects = []

        # For now, return empty list as we don't have a simple way to process PointCloud2 in Python
        # without additional libraries like open3d or pcl
        return detected_objects

    def distance_to_cluster(self, point, cluster):
        """Calculate distance from point to cluster of points"""
        if not cluster:
            return float('inf')

        distances = [math.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in cluster]
        return min(distances) if distances else float('inf')

    def update_tracked_objects(self, detected_objects):
        """Update tracked objects with new detections"""
        if not self.tracking_enabled:
            self.tracked_objects = {obj.id: obj for obj in detected_objects}
            return

        # Simple tracking by matching positions
        for new_obj in detected_objects:
            matched = False
            for tracked_id, tracked_obj in self.tracked_objects.items():
                # Calculate distance between new detection and tracked object
                dist = math.sqrt(
                    (new_obj.position.x - tracked_obj.position.x)**2 +
                    (new_obj.position.y - tracked_obj.position.y)**2
                )

                if dist < 0.5:  # 50cm threshold for matching
                    # Update tracked object with new information
                    self.tracked_objects[tracked_id] = new_obj
                    matched = True
                    break

            if not matched:
                # New object detected
                new_id = f"tracked_{self.next_object_id}"
                self.next_object_id += 1
                new_obj.id = new_id
                self.tracked_objects[new_id] = new_obj

        # Remove old objects that haven't been seen recently
        current_time = time.time()
        objects_to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if current_time - obj.timestamp > 2.0:  # Remove objects not seen in 2 seconds
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]

    def publish_detected_objects(self):
        """Publish detected objects as PoseArray"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'perception_frame'

        for obj_id, obj in self.tracked_objects.items():
            pose = Pose()
            pose.position = obj.position
            # For now, set orientation to identity
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.object_detection_publisher.publish(pose_array)

    def publish_visualization(self):
        """Publish visualization markers for detected objects"""
        marker_array = MarkerArray()
        marker_id = 0

        for obj_id, obj in self.tracked_objects.items():
            # Create a marker for the object
            marker = Marker()
            marker.header.frame_id = "perception_frame"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position
            marker.pose.position = obj.position
            marker.pose.orientation.w = 1.0

            # Scale based on detected size
            marker.scale.x = max(0.1, obj.size.x)
            marker.scale.y = max(0.1, obj.size.y)
            marker.scale.z = max(0.1, obj.size.z)

            # Color based on object type
            if "red" in obj.type:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif "blue" in obj.type:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0  # Yellow for unknown types

            marker.color.a = obj.confidence  # Transparency based on confidence

            marker_array.markers.append(marker)

        self.visualization_publisher.publish(marker_array)

    def enable_perception(self, enable: bool):
        """Enable or disable perception processing"""
        self.perception_enabled = enable
        self.get_logger().info(f"Perception processing {'enabled' if enable else 'disabled'}")

    def set_detection_threshold(self, threshold: float):
        """Set the object detection confidence threshold"""
        self.object_detection_threshold = threshold
        self.get_logger().info(f"Detection threshold set to {threshold}")

    def get_tracked_objects(self) -> Dict[str, DetectedObject]:
        """Get currently tracked objects"""
        return self.tracked_objects.copy()

    def get_perception_stats(self) -> Dict[str, float]:
        """Get perception statistics"""
        return {
            'object_count': len(self.tracked_objects),
            'last_update_time': self.last_perception_time,
            'enabled': self.perception_enabled
        }


def main(args=None):
    rclpy.init(args=args)

    perception_node = PerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()