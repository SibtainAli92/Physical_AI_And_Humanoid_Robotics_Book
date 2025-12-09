#!/usr/bin/env python3
# slam_system.py
# SLAM system for mapping and localization

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan, Image, PointCloud2, Imu
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion
from tf2_ros import TransformBroadcaster, TransformStamped
from std_msgs.msg import Header, Float64
import numpy as np
import cv2
from cv_bridge import CvBridge
from typing import List, Tuple, Optional, Dict
import threading
import math
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass


@dataclass
class MapCell:
    """Data class for map cells"""
    occupancy: float  # -1: unknown, 0: free, 100: occupied
    last_seen: float  # timestamp


class SLAMSystem(Node):
    """
    SLAM system for mapping and localization
    """
    def __init__(self):
        super().__init__('slam_system')

        # Declare parameters
        self.declare_parameter('map_resolution', 0.05)  # meters per cell
        self.declare_parameter('map_width', 20.0)      # meters
        self.declare_parameter('map_height', 20.0)     # meters
        self.declare_parameter('map_origin_x', -10.0)  # meters
        self.declare_parameter('map_origin_y', -10.0)  # meters
        self.declare_parameter('laser_max_range', 10.0)  # meters
        self.declare_parameter('update_rate', 5.0)     # Hz
        self.declare_parameter('odom_alpha1', 0.01)    # rotation noise
        self.declare_parameter('odom_alpha2', 0.01)    # translation noise
        self.declare_parameter('odom_alpha3', 0.01)    # rotation noise
        self.declare_parameter('odom_alpha4', 0.01)    # translation noise

        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.map_origin_x = self.get_parameter('map_origin_x').value
        self.map_origin_y = self.get_parameter('map_origin_y').value
        self.laser_max_range = self.get_parameter('laser_max_range').value
        self.update_rate = self.get_parameter('update_rate').value
        self.odom_alpha1 = self.get_parameter('odom_alpha1').value
        self.odom_alpha2 = self.get_parameter('odom_alpha2').value
        self.odom_alpha3 = self.get_parameter('odom_alpha3').value
        self.odom_alpha4 = self.get_parameter('odom_alpha4').value

        # Initialize map dimensions
        self.map_width_cells = int(self.map_width / self.map_resolution)
        self.map_height_cells = int(self.map_height / self.map_resolution)

        # Initialize map
        self.occupancy_map = np.full((self.map_height_cells, self.map_width_cells), -1, dtype=np.int8)  # -1 = unknown
        self.last_map_update = np.zeros((self.map_height_cells, self.map_width_cells))

        # Initialize pose
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_odom = np.array([0.0, 0.0, 0.0])  # x, y, theta for odometry

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            QoSProfile(depth=1)
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/slam/pose',
            QoSProfile(depth=10)
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            '/slam/odometry',
            QoSProfile(depth=10)
        )

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.scan_callback,
            QoSProfile(depth=5)
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            QoSProfile(depth=10)
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            QoSProfile(depth=10)
        )

        # Initialize timing
        self.last_scan_time = self.get_clock().now()
        self.map_update_timer = self.create_timer(1.0/self.update_rate, self.publish_map)

        # Internal state
        self.processing_lock = threading.Lock()
        self.odom_queue = []  # Queue for odometry data
        self.scan_queue = []  # Queue for scan data

        self.get_logger().info('SLAM System initialized')

    def odom_callback(self, msg: Odometry):
        """
        Callback for odometry messages
        """
        with self.processing_lock:
            # Extract pose from odometry message
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y

            # Convert quaternion to euler
            quat = msg.pose.pose.orientation
            r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            euler = r.as_euler('xyz')
            theta = euler[2]  # Only need yaw

            self.current_odom = np.array([x, y, theta])

    def imu_callback(self, msg: Imu):
        """
        Callback for IMU messages (for additional orientation data)
        """
        # In a real implementation, IMU data would be fused with odometry
        pass

    def scan_callback(self, msg: LaserScan):
        """
        Callback for laser scan messages
        """
        with self.processing_lock:
            # Convert laser scan to occupancy map
            self.update_map_from_scan(msg)

            # Update robot pose based on odometry and scan matching
            self.update_pose_from_scan(msg)

            self.last_scan_time = self.get_clock().now()

    def update_map_from_scan(self, scan_msg: LaserScan):
        """
        Update occupancy map based on laser scan
        """
        # Get robot's current position in map coordinates
        robot_x = int((self.current_pose[0] - self.map_origin_x) / self.map_resolution)
        robot_y = int((self.current_pose[1] - self.map_origin_y) / self.map_resolution)

        # Check if robot position is within map bounds
        if (0 <= robot_x < self.map_width_cells and 0 <= robot_y < self.map_height_cells):
            # Process each range measurement
            for i, range_val in enumerate(scan_msg.ranges):
                if not (math.isnan(range_val) or math.isinf(range_val)) and range_val <= self.laser_max_range:
                    # Calculate angle of this measurement
                    angle = scan_msg.angle_min + i * scan_msg.angle_increment + self.current_pose[2]

                    # Calculate end point of this measurement in world coordinates
                    end_x = self.current_pose[0] + range_val * math.cos(angle)
                    end_y = self.current_pose[1] + range_val * math.sin(angle)

                    # Convert to map coordinates
                    map_x = int((end_x - self.map_origin_x) / self.map_resolution)
                    map_y = int((end_y - self.map_origin_y) / self.map_resolution)

                    # Check bounds
                    if (0 <= map_x < self.map_width_cells and 0 <= map_y < self.map_height_cells):
                        # Mark endpoint as occupied
                        self.occupancy_map[map_y, map_x] = 100  # occupied

                        # Ray tracing: mark free space along the beam
                        self.ray_trace(robot_x, robot_y, map_x, map_y)

    def ray_trace(self, x0: int, y0: int, x1: int, y1: int):
        """
        Ray tracing to mark free space between robot and obstacle
        """
        # Bresenham's line algorithm to mark free space
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            # Check bounds
            if not (0 <= x < self.map_width_cells and 0 <= y < self.map_height_cells):
                break

            # Don't mark endpoint as free (that's the obstacle)
            if x == x1 and y == y1:
                break

            # Mark as free space (with probability - using simple model)
            # Only mark as free if not already occupied
            if self.occupancy_map[y, x] != 100:
                self.occupancy_map[y, x] = 0

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def update_pose_from_scan(self, scan_msg: LaserScan):
        """
        Update robot pose using scan matching (simplified implementation)
        """
        # In a real implementation, this would perform scan matching
        # For now, we'll just use odometry with some noise modeling
        with self.processing_lock:
            # This is a simplified pose update - in reality you'd use ICP or other scan matching
            # algorithms to correct the pose based on observed features
            pass

    def publish_map(self):
        """
        Publish the occupancy grid map
        """
        if self.occupancy_map is not None:
            # Create occupancy grid message
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            # Set map info
            msg.info.resolution = self.map_resolution
            msg.info.width = self.map_width_cells
            msg.info.height = self.map_height_cells
            msg.info.origin.position.x = self.map_origin_x
            msg.info.origin.position.y = self.map_origin_y
            msg.info.origin.position.z = 0.0
            msg.info.origin.orientation.x = 0.0
            msg.info.origin.orientation.y = 0.0
            msg.info.origin.orientation.z = 0.0
            msg.info.origin.orientation.w = 1.0

            # Flatten map data for message
            map_data = self.occupancy_map.flatten().tolist()
            msg.data = map_data

            self.map_pub.publish(msg)

    def publish_pose(self):
        """
        Publish current estimated pose
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = float(self.current_pose[0])
        pose_msg.pose.position.y = float(self.current_pose[1])
        pose_msg.pose.position.z = 0.0

        # Convert angle to quaternion
        quat = R.from_euler('z', self.current_pose[2]).as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        t.transform.translation.x = float(self.current_pose[0])
        t.transform.translation.y = float(self.current_pose[1])
        t.transform.translation.z = 0.0

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)


class SemanticSegmentation(Node):
    """
    Semantic segmentation for environment understanding
    """
    def __init__(self):
        super().__init__('semantic_segmentation')

        # Declare parameters
        self.declare_parameter('segmentation_threshold', 0.5)
        self.declare_parameter('enable_instance_segmentation', False)

        self.segmentation_threshold = self.get_parameter('segmentation_threshold').value
        self.enable_instance_segmentation = self.get_parameter('enable_instance_segmentation').value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
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

        self.get_logger().info('Semantic Segmentation initialized')

    def image_callback(self, msg: Image):
        """
        Callback for image messages to perform semantic segmentation
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform segmentation (simulated implementation)
            segmented_image = self.perform_segmentation(cv_image)

            # Publish segmented image
            segmented_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='mono8')
            segmented_msg.header = msg.header
            self.segmentation_pub.publish(segmented_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image for segmentation: {str(e)}')

    def perform_segmentation(self, cv_image: np.ndarray) -> np.ndarray:
        """
        Perform semantic segmentation on the image
        This is a simulated implementation
        """
        # In a real implementation, this would use a deep learning model
        # For simulation, we'll use simple color-based segmentation

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects/classes
        # These are example ranges - in reality, you'd have trained classes
        lower_floor = np.array([0, 0, 100])
        upper_floor = np.array([180, 50, 255])
        mask_floor = cv2.inRange(hsv, lower_floor, upper_floor)

        lower_obstacle = np.array([0, 50, 50])
        upper_obstacle = np.array([20, 255, 255])
        mask_obstacle = cv2.inRange(hsv, lower_obstacle, upper_obstacle)

        # Combine masks
        combined_mask = np.zeros_like(mask_floor)
        combined_mask[mask_floor > 0] = 50   # Floor class
        combined_mask[mask_obstacle > 0] = 100  # Obstacle class

        # In a real implementation, this would return class probabilities for each pixel
        return combined_mask


def main(args=None):
    rclpy.init(args=args)

    # Create SLAM nodes
    slam_system = SLAMSystem()
    segmentation = SemanticSegmentation()

    # Use a MultiThreadedExecutor to handle callbacks from multiple nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(slam_system)
    executor.add_node(segmentation)

    try:
        executor.spin()
    except KeyboardInterrupt:
        slam_system.get_logger().info('SLAM system interrupted by user')
        segmentation.get_logger().info('Semantic segmentation interrupted by user')
    finally:
        slam_system.destroy_node()
        segmentation.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()