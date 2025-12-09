#!/usr/bin/env python3
# computer_vision_pipeline.py
# Computer vision pipeline for humanoid robot perception

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
import cv2
import numpy as np
from cv_bridge import CvBridge
from typing import List, Tuple, Optional
import threading
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Data class for detection results"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[float, float]       # x, y center coordinates
    pose: Optional[Point] = None      # 3D pose if available


class ComputerVisionPipeline(Node):
    """
    Computer vision pipeline for object detection and recognition
    """
    def __init__(self):
        super().__init__('computer_vision_pipeline')

        # Declare parameters
        self.declare_parameter('detection_threshold', 0.5)
        self.declare_parameter('max_detection_objects', 10)
        self.declare_parameter('enable_segmentation', False)
        self.declare_parameter('enable_pose_estimation', False)

        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.max_detection_objects = self.get_parameter('max_detection_objects').value
        self.enable_segmentation = self.get_parameter('enable_segmentation').value
        self.enable_pose_estimation = self.get_parameter('enable_pose_estimation').value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/perception/objects',
            QoSProfile(depth=10)
        )

        self.segmentation_pub = self.create_publisher(
            Image,
            '/perception/semantic_segmentation',
            QoSProfile(depth=10)
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            QoSProfile(depth=1)
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            QoSProfile(depth=10)
        )

        # Internal state
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.processing_lock = threading.Lock()
        self.last_image = None

        # Initialize detection models (simulated - in real implementation, you'd load actual models)
        self.initialize_models()

        self.get_logger().info('Computer Vision Pipeline initialized')

    def initialize_models(self):
        """
        Initialize computer vision models
        In a real implementation, this would load actual ML models
        """
        # Simulated model initialization
        self.get_logger().info('Simulated model initialization completed')

    def camera_info_callback(self, msg: CameraInfo):
        """
        Callback for camera info messages
        """
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg: Image):
        """
        Callback for image messages
        """
        with self.processing_lock:
            try:
                # Convert ROS image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Process the image
                detections = self.process_image(cv_image)

                # Publish detections
                self.publish_detections(detections, msg.header)

                # Store for potential visualization
                self.last_image = cv_image

            except Exception as e:
                self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, cv_image: np.ndarray) -> List[DetectionResult]:
        """
        Process image and return detections
        This is a simplified implementation - in real system, this would use ML models
        """
        detections = []

        # Simulate object detection using basic computer vision techniques
        # In a real implementation, this would use a trained ML model
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect objects using simple shape detection (for demonstration)
        # In reality, this would use a neural network for object detection
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center
                center_x = x + w / 2
                center_y = y + h / 2

                # Create detection result
                detection = DetectionResult(
                    class_name="object",  # In real implementation, this would be actual class
                    confidence=min(0.9, area / 10000),  # Simulated confidence
                    bbox=(x, y, w, h),
                    center=(center_x, center_y)
                )

                detections.append(detection)

        # Sort by confidence and limit to max detections
        detections.sort(key=lambda x: x.confidence, reverse=True)
        return detections[:self.max_detection_objects]

    def publish_detections(self, detections: List[DetectionResult], header: Header):
        """
        Publish detection results to ROS topic
        """
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            if detection.confidence >= self.detection_threshold:
                detection_2d = Detection2D()

                # Set ID and confidence
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = detection.class_name
                hypothesis.score = detection.confidence
                detection_2d.results.append(hypothesis)

                # Set bounding box
                detection_2d.bbox.center.x = detection.center[0]
                detection_2d.bbox.center.y = detection.center[1]
                detection_2d.bbox.size_x = detection.bbox[2]
                detection_2d.bbox.size_y = detection.bbox[3]

                detection_array.detections.append(detection_2d)

        self.detection_pub.publish(detection_array)

    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get camera intrinsics for 3D pose estimation
        """
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return self.camera_matrix, self.distortion_coeffs
        return None


class HumanPoseEstimator(Node):
    """
    Human pose estimation for interaction
    """
    def __init__(self):
        super().__init__('human_pose_estimator')

        # Declare parameters
        self.declare_parameter('pose_confidence_threshold', 0.7)
        self.declare_parameter('enable_3d_pose', False)

        self.pose_confidence_threshold = self.get_parameter('pose_confidence_threshold').value
        self.enable_3d_pose = self.get_parameter('enable_3d_pose').value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.pose_pub = self.create_publisher(
            Detection2DArray,  # Using Detection2DArray for pose data
            '/perception/human_poses',
            QoSProfile(depth=10)
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            QoSProfile(depth=1)
        )

        self.get_logger().info('Human Pose Estimator initialized')

    def image_callback(self, msg: Image):
        """
        Callback for image messages to estimate human poses
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Estimate poses (simulated implementation)
            poses = self.estimate_poses(cv_image)

            # Publish poses
            self.publish_poses(poses, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image for pose estimation: {str(e)}')

    def estimate_poses(self, cv_image: np.ndarray) -> List[DetectionResult]:
        """
        Estimate human poses in the image
        This is a simulated implementation
        """
        poses = []

        # Simulate pose detection using basic techniques
        # In reality, this would use a pose estimation model like OpenPose or MediaPipe
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Use Haar cascades for person detection as a simple example
        # In real implementation, use a proper pose estimation model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Calculate center
            center_x = x + w / 2
            center_y = y + h / 2

            # Create pose result
            pose = DetectionResult(
                class_name="person",
                confidence=0.8,  # Simulated confidence
                bbox=(x, y, w, h),
                center=(center_x, center_y)
            )

            poses.append(pose)

        return poses

    def publish_poses(self, poses: List[DetectionResult], header: Header):
        """
        Publish pose estimation results
        """
        pose_array = Detection2DArray()
        pose_array.header = header

        for pose in poses:
            if pose.confidence >= self.pose_confidence_threshold:
                pose_2d = Detection2D()

                # Set ID and confidence
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = pose.class_name
                hypothesis.score = pose.confidence
                pose_2d.results.append(hypothesis)

                # Set bounding box
                pose_2d.bbox.center.x = pose.center[0]
                pose_2d.bbox.center.y = pose.center[1]
                pose_2d.bbox.size_x = pose.bbox[2]
                pose_2d.bbox.size_y = pose.bbox[3]

                pose_array.detections.append(pose_2d)

        self.pose_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)

    # Create perception nodes
    cv_pipeline = ComputerVisionPipeline()
    pose_estimator = HumanPoseEstimator()

    # Use a MultiThreadedExecutor to handle callbacks from multiple nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(cv_pipeline)
    executor.add_node(pose_estimator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        cv_pipeline.get_logger().info('Computer vision pipeline interrupted by user')
        pose_estimator.get_logger().info('Pose estimator interrupted by user')
    finally:
        cv_pipeline.destroy_node()
        pose_estimator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()