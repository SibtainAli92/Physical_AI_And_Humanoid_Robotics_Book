#!/usr/bin/env python3

"""
Sensor Simulator Node for Humanoid Robot Digital Twin

This node simulates various sensors of the humanoid robot,
generating realistic sensor data based on the simulated environment
and robot state.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Header
from sensor_msgs.msg import JointState, Imu, LaserScan, PointCloud2, PointField
from geometry_msgs.msg import Vector3, Point, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import time
import math
import random
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SensorSpec:
    """Specification for a simulated sensor"""
    name: str
    topic: str
    update_rate: float
    noise_level: float
    last_update: float


class SensorSimulatorNode(Node):
    def __init__(self):
        super().__init__('sensor_simulator_node')

        # Publishers for simulated sensor data
        self.joint_state_publisher = self.create_publisher(JointState, 'sim/joint_states', 10)
        self.imu_publisher = self.create_publisher(Imu, 'sim/imu/data', 10)
        self.laser_scan_publisher = self.create_publisher(LaserScan, 'sim/scan', 10)
        self.odom_publisher = self.create_publisher(Odometry, 'sim/odom', 10)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, 'sim/point_cloud', 10)
        self.camera_publisher = self.create_publisher(String, 'sim/camera/image', 10)  # Using String as placeholder
        self.force_torque_publisher = self.create_publisher(Wrench, 'sim/force_torque', 10)

        # Subscribers for simulation state
        self.sim_robot_state_subscriber = self.create_subscription(
            JointState, 'sim/joint_states', self.sim_robot_state_callback, 10)
        self.sim_odom_subscriber = self.create_subscription(
            Odometry, 'sim/odom', self.sim_odom_callback, 10)

        # Timer for sensor simulation
        self.sensor_timer = self.create_timer(0.01, self.sensor_simulation_callback)  # 100 Hz

        # Sensor specifications
        self.sensors = {
            'imu': SensorSpec('imu', 'sim/imu/data', 100.0, 0.02, 0.0),  # 100 Hz, low noise
            'lidar': SensorSpec('lidar', 'sim/scan', 10.0, 0.01, 0.0),   # 10 Hz, very low noise
            'joint_encoders': SensorSpec('joint_encoders', 'sim/joint_states', 50.0, 0.001, 0.0),  # 50 Hz, very low noise
            'force_torque': SensorSpec('force_torque', 'sim/force_torque', 200.0, 0.05, 0.0),  # 200 Hz, moderate noise
            'camera': SensorSpec('camera', 'sim/camera/image', 30.0, 0.0, 0.0),  # 30 Hz, no noise (image)
            'point_cloud': SensorSpec('point_cloud', 'sim/point_cloud', 15.0, 0.01, 0.0),  # 15 Hz, low noise
        }

        # Robot state tracking
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_position = Point(x=0.0, y=0.0, z=0.0)
        self.current_orientation = Vector3(x=0.0, y=0.0, z=0.0)
        self.current_linear_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        self.current_angular_velocity = Vector3(x=0.0, y=0.0, z=0.0)

        # Environment state for sensor simulation
        self.environment_objects = [
            {'type': 'wall', 'position': Point(x=2.0, y=0.0, z=1.0), 'size': [0.1, 4.0, 2.0]},
            {'type': 'box', 'position': Point(x=-1.0, y=1.0, z=0.5), 'size': [0.5, 0.5, 1.0]},
            {'type': 'cylinder', 'position': Point(x=0.0, y=-1.5, z=0.75), 'size': [0.3, 1.5]},  # radius, height
        ]

        # Sensor simulation parameters
        self.sim_time = time.time()
        self.gravity = -9.81
        self.robot_dimensions = {'height': 1.0, 'width': 0.5, 'depth': 0.3}

        self.get_logger().info('Sensor Simulator Node initialized')

    def sim_robot_state_callback(self, msg):
        """Callback for simulated robot state updates"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def sim_odom_callback(self, msg):
        """Callback for simulated odometry updates"""
        self.current_position = msg.pose.pose.position
        # For simplicity, using orientation vector instead of quaternion
        self.current_orientation.x = msg.pose.pose.orientation.x
        self.current_orientation.y = msg.pose.pose.orientation.y
        self.current_orientation.z = msg.pose.pose.orientation.z
        self.current_linear_velocity = msg.twist.twist.linear
        self.current_angular_velocity = msg.twist.twist.angular

    def sensor_simulation_callback(self):
        """Main sensor simulation callback"""
        current_time = time.time()
        dt = current_time - self.sim_time
        self.sim_time = current_time

        # Update each sensor based on its update rate
        for sensor_name, spec in self.sensors.items():
            if current_time - spec.last_update >= 1.0 / spec.update_rate:
                self.simulate_sensor(sensor_name, current_time)
                spec.last_update = current_time

    def simulate_sensor(self, sensor_name: str, current_time: float):
        """Simulate a specific sensor"""
        if sensor_name == 'joint_encoders':
            self.simulate_joint_encoders()
        elif sensor_name == 'imu':
            self.simulate_imu()
        elif sensor_name == 'lidar':
            self.simulate_lidar()
        elif sensor_name == 'force_torque':
            self.simulate_force_torque()
        elif sensor_name == 'camera':
            self.simulate_camera()
        elif sensor_name == 'point_cloud':
            self.simulate_point_cloud()

    def simulate_joint_encoders(self):
        """Simulate joint encoder data"""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = 'sim_base_link'

        # Add some noise to the actual positions
        for joint_name, position in self.current_joint_positions.items():
            noisy_position = position + random.gauss(0, self.sensors['joint_encoders'].noise_level)
            joint_state_msg.name.append(joint_name)
            joint_state_msg.position.append(noisy_position)

        # Add velocities with noise
        for joint_name, velocity in self.current_joint_velocities.items():
            noisy_velocity = velocity + random.gauss(0, self.sensors['joint_encoders'].noise_level * 0.1)
            joint_state_msg.velocity.append(noisy_velocity)
            joint_state_msg.effort.append(0.0)  # Simplified

        self.joint_state_publisher.publish(joint_state_msg)

    def simulate_imu(self):
        """Simulate IMU data"""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'sim_imu_link'

        # Simulate linear acceleration with gravity and motion
        imu_msg.linear_acceleration.x = self.current_linear_velocity.x * 10 + random.gauss(0, self.sensors['imu'].noise_level)
        imu_msg.linear_acceleration.y = self.current_linear_velocity.y * 10 + random.gauss(0, self.sensors['imu'].noise_level)
        imu_msg.linear_acceleration.z = self.current_linear_velocity.z * 10 + self.gravity + random.gauss(0, self.sensors['imu'].noise_level)

        # Simulate angular velocity
        imu_msg.angular_velocity.x = self.current_angular_velocity.x + random.gauss(0, self.sensors['imu'].noise_level)
        imu_msg.angular_velocity.y = self.current_angular_velocity.y + random.gauss(0, self.sensors['imu'].noise_level)
        imu_msg.angular_velocity.z = self.current_angular_velocity.z + random.gauss(0, self.sensors['imu'].noise_level)

        # Simulate orientation (integrate angular velocity)
        # Simplified - in reality would use proper quaternion integration
        imu_msg.orientation.w = 1.0  # Simplified quaternion
        imu_msg.orientation.x = self.current_orientation.x + random.gauss(0, self.sensors['imu'].noise_level)
        imu_msg.orientation.y = self.current_orientation.y + random.gauss(0, self.sensors['imu'].noise_level)
        imu_msg.orientation.z = self.current_orientation.z + random.gauss(0, self.sensors['imu'].noise_level)

        self.imu_publisher.publish(imu_msg)

    def simulate_lidar(self):
        """Simulate LIDAR data"""
        laser_msg = LaserScan()
        laser_msg.header.stamp = self.get_clock().now().to_msg()
        laser_msg.header.frame_id = 'sim_laser_link'
        laser_msg.angle_min = -math.pi / 2
        laser_msg.angle_max = math.pi / 2
        laser_msg.angle_increment = math.pi / 180  # 1 degree
        laser_msg.time_increment = 0.0
        laser_msg.scan_time = 0.1
        laser_msg.range_min = 0.1
        laser_msg.range_max = 10.0

        # Calculate ranges based on environment objects
        num_ranges = int((laser_msg.angle_max - laser_msg.angle_min) / laser_msg.angle_increment) + 1
        ranges = []

        for i in range(num_ranges):
            angle = laser_msg.angle_min + i * laser_msg.angle_increment

            # Calculate ray direction in global frame
            ray_dir_x = math.cos(angle)
            ray_dir_y = math.sin(angle)

            # Transform ray to robot's local frame
            robot_x, robot_y = self.current_position.x, self.current_position.y
            ray_start_x, ray_start_y = robot_x, robot_y

            # Find closest intersection with environment objects
            min_range = laser_msg.range_max

            for obj in self.environment_objects:
                if obj['type'] == 'wall':
                    # Check intersection with wall (simplified as infinite plane)
                    wall_x = obj['position'].x
                    if ray_dir_x != 0:  # Not parallel to wall
                        t = (wall_x - ray_start_x) / ray_dir_x
                        if t > 0:  # Intersection in front
                            intersect_y = ray_start_y + t * ray_dir_y
                            # Check if intersection is within wall bounds
                            wall_half_height = obj['size'][1] / 2
                            if abs(intersect_y - obj['position'].y) <= wall_half_height:
                                range_val = abs(t)
                                if range_val < min_range:
                                    min_range = range_val
                elif obj['type'] == 'box':
                    # Simplified box intersection
                    box_x, box_y = obj['position'].x, obj['position'].y
                    box_half_size_x, box_half_size_y = obj['size'][0] / 2, obj['size'][1] / 2

                    # Check intersection with box boundaries
                    if ray_dir_x != 0:
                        t1 = (box_x - box_half_size_x - ray_start_x) / ray_dir_x
                        t2 = (box_x + box_half_size_x - ray_start_x) / ray_dir_x
                        for t in [t1, t2]:
                            if t > 0:
                                intersect_y = ray_start_y + t * ray_dir_y
                                if (abs(intersect_y - box_y) <= box_half_size_y):
                                    range_val = abs(t)
                                    if range_val < min_range:
                                        min_range = range_val

                    if ray_dir_y != 0:
                        t1 = (box_y - box_half_size_y - ray_start_y) / ray_dir_y
                        t2 = (box_y + box_half_size_y - ray_start_y) / ray_dir_y
                        for t in [t1, t2]:
                            if t > 0:
                                intersect_x = ray_start_x + t * ray_dir_x
                                if (abs(intersect_x - box_x) <= box_half_size_x):
                                    range_val = math.sqrt((intersect_x - ray_start_x)**2 + (intersect_y - ray_start_y)**2)
                                    if range_val < min_range:
                                        min_range = range_val

            # Add noise to the range
            noisy_range = min_range + random.gauss(0, self.sensors['lidar'].noise_level)
            # Ensure range is within valid bounds
            noisy_range = max(laser_msg.range_min, min(laser_msg.range_max, noisy_range))
            ranges.append(noisy_range)

        laser_msg.ranges = ranges
        laser_msg.intensities = [100.0] * len(ranges)  # Simplified intensity

        self.laser_scan_publisher.publish(laser_msg)

    def simulate_force_torque(self):
        """Simulate force/torque sensor data"""
        from geometry_msgs.msg import Wrench
        wrench_msg = Wrench()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'sim_force_torque_link'

        # Simulate forces based on robot motion and environment interaction
        wrench_msg.force.x = self.current_linear_velocity.x * 5 + random.gauss(0, self.sensors['force_torque'].noise_level)
        wrench_msg.force.y = self.current_linear_velocity.y * 5 + random.gauss(0, self.sensors['force_torque'].noise_level)
        wrench_msg.force.z = self.current_linear_velocity.z * 5 + 50 + random.gauss(0, self.sensors['force_torque'].noise_level)  # Weight component

        # Simulate torques
        wrench_msg.torque.x = self.current_angular_velocity.x * 2 + random.gauss(0, self.sensors['force_torque'].noise_level)
        wrench_msg.torque.y = self.current_angular_velocity.y * 2 + random.gauss(0, self.sensors['force_torque'].noise_level)
        wrench_msg.torque.z = self.current_angular_velocity.z * 2 + random.gauss(0, self.sensors['force_torque'].noise_level)

        self.force_torque_publisher.publish(wrench_msg)

    def simulate_camera(self):
        """Simulate camera data (simplified as metadata string)"""
        # In a real system, this would publish actual image data
        # For now, we'll publish a string with simulated image information
        camera_msg = String()
        camera_msg.data = f"simulated_image_{int(self.sim_time * 30)}.jpg,position=({self.current_position.x:.2f},{self.current_position.y:.2f},{self.current_position.z:.2f}),timestamp={self.sim_time:.3f}"
        self.camera_publisher.publish(camera_msg)

    def simulate_point_cloud(self):
        """Simulate point cloud data"""
        # Create a simplified point cloud message
        # This is a simplified version - a real implementation would be more complex
        pc_msg = PointCloud2()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = 'sim_camera_link'
        pc_msg.height = 1
        pc_msg.width = 0  # Will be set after points are generated
        pc_msg.is_dense = False
        pc_msg.is_bigendian = False

        # Define point fields (x, y, z)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.fields = fields
        pc_msg.point_step = 12  # 3 floats * 4 bytes each

        # Generate points based on environment objects
        points_data = []
        for obj in self.environment_objects:
            if obj['type'] == 'wall':
                # Generate points along the wall
                for i in range(20):  # 20 points along the wall
                    for j in range(10):  # 10 points in height
                        x = obj['position'].x
                        y = obj['position'].y - obj['size'][1]/2 + i * obj['size'][1]/19
                        z = obj['position'].z - obj['size'][2]/2 + j * obj['size'][2]/9
                        # Add small noise
                        x += random.gauss(0, 0.01)
                        y += random.gauss(0, 0.01)
                        z += random.gauss(0, 0.01)
                        points_data.extend([x, y, z])
            elif obj['type'] == 'box':
                # Generate points on the surface of the box
                # Bottom face
                for _ in range(50):
                    x = obj['position'].x - obj['size'][0]/2 + random.random() * obj['size'][0]
                    y = obj['position'].y - obj['size'][1]/2 + random.random() * obj['size'][1]
                    z = obj['position'].z - obj['size'][2]/2
                    points_data.extend([x, y, z])
                # Top face
                for _ in range(50):
                    x = obj['position'].x - obj['size'][0]/2 + random.random() * obj['size'][0]
                    y = obj['position'].y - obj['size'][1]/2 + random.random() * obj['size'][1]
                    z = obj['position'].z + obj['size'][2]/2
                    points_data.extend([x, y, z])
                # Side faces (simplified)
                for _ in range(100):
                    face = random.randint(0, 3)
                    if face == 0:  # Front
                        x = obj['position'].x - obj['size'][0]/2
                        y = obj['position'].y - obj['size'][1]/2 + random.random() * obj['size'][1]
                        z = obj['position'].z - obj['size'][2]/2 + random.random() * obj['size'][2]
                    elif face == 1:  # Back
                        x = obj['position'].x + obj['size'][0]/2
                        y = obj['position'].y - obj['size'][1]/2 + random.random() * obj['size'][1]
                        z = obj['position'].z - obj['size'][2]/2 + random.random() * obj['size'][2]
                    elif face == 2:  # Left
                        x = obj['position'].x - obj['size'][0]/2 + random.random() * obj['size'][0]
                        y = obj['position'].y - obj['size'][1]/2
                        z = obj['position'].z - obj['size'][2]/2 + random.random() * obj['size'][2]
                    else:  # Right
                        x = obj['position'].x - obj['size'][0]/2 + random.random() * obj['size'][0]
                        y = obj['position'].y + obj['size'][1]/2
                        z = obj['position'].z - obj['size'][2]/2 + random.random() * obj['size'][2]
                    points_data.extend([x, y, z])

        # Convert points to binary data
        import struct
        pc_data = []
        for x, y, z in zip(points_data[::3], points_data[1::3], points_data[2::3]):
            # Pack as binary float data
            pc_data.extend(struct.pack('fff', x, y, z))

        pc_msg.data = bytes(pc_data)
        pc_msg.width = len(points_data) // 3
        self.point_cloud_publisher.publish(pc_msg)

    def add_environment_object(self, obj_type: str, position: Point, size: List[float]):
        """Add an object to the simulated environment"""
        obj = {
            'type': obj_type,
            'position': position,
            'size': size
        }
        self.environment_objects.append(obj)
        self.get_logger().info(f'Added {obj_type} at ({position.x}, {position.y}, {position.z})')


def main(args=None):
    rclpy.init(args=args)

    sensor_simulator_node = SensorSimulatorNode()

    try:
        rclpy.spin(sensor_simulator_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_simulator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()