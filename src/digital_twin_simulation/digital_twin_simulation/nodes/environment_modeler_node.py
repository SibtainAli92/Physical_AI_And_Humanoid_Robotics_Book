#!/usr/bin/env python3

"""
Environment Modeler Node for Humanoid Robot Digital Twin

This node creates and manages the virtual environment for the digital twin,
including static and dynamic objects, terrain modeling, and environmental
conditions.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point, Vector3, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
import time
import math
import random
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class EnvironmentObject:
    """Represents an object in the simulated environment"""
    id: str
    type: str  # 'static', 'dynamic', 'decoration'
    shape: str  # 'box', 'sphere', 'cylinder', 'mesh'
    position: Point
    orientation: Vector3
    size: List[float]  # [width, depth, height] for box, [radius, height] for cylinder
    color: Tuple[float, float, float]  # RGB values (0-1)
    dynamic_properties: Dict = None  # Properties for dynamic objects


class EnvironmentModelerNode(Node):
    def __init__(self):
        super().__init__('environment_modeler_node')

        # Publishers for environment data
        self.environment_publisher = self.create_publisher(MarkerArray, 'sim/environment', 10)
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, 'sim/map', 10)
        self.static_objects_publisher = self.create_publisher(PoseArray, 'sim/static_objects', 10)
        self.environment_status_publisher = self.create_publisher(String, 'sim/environment_status', 10)

        # Subscribers for environment updates
        self.add_object_subscriber = self.create_subscription(
            String, 'sim/add_object', self.add_object_callback, 10)
        self.remove_object_subscriber = self.create_subscription(
            String, 'sim/remove_object', self.remove_object_callback, 10)

        # Timer for environment updates
        self.environment_timer = self.create_timer(0.5, self.environment_update_callback)  # 2 Hz

        # Environment state
        self.environment_objects: Dict[str, EnvironmentObject] = {}
        self.terrain_map = None
        self.environment_bounds = {
            'min_x': -10.0, 'max_x': 10.0,
            'min_y': -10.0, 'max_y': 10.0,
            'min_z': -1.0, 'max_z': 3.0
        }
        self.sim_time = time.time()
        self.last_update_time = time.time()

        # Initialize with default environment
        self.initialize_default_environment()

        # Grid map parameters
        self.grid_resolution = 0.1  # 10cm resolution
        self.grid_width = int((self.environment_bounds['max_x'] - self.environment_bounds['min_x']) / self.grid_resolution)
        self.grid_height = int((self.environment_bounds['max_y'] - self.environment_bounds['min_y']) / self.grid_resolution)

        self.get_logger().info('Environment Modeler Node initialized')

    def initialize_default_environment(self):
        """Initialize the environment with default objects"""
        # Add ground plane
        ground_obj = EnvironmentObject(
            id='ground_plane',
            type='static',
            shape='box',
            position=Point(x=0.0, y=0.0, z=-0.5),
            orientation=Vector3(x=0.0, y=0.0, z=0.0),
            size=[20.0, 20.0, 1.0],  # Large flat surface
            color=(0.6, 0.6, 0.6)  # Gray
        )
        self.environment_objects['ground_plane'] = ground_obj

        # Add some default obstacles
        obstacles = [
            {'id': 'wall_1', 'pos': Point(x=5.0, y=0.0, z=1.0), 'size': [0.2, 8.0, 2.0], 'color': (0.8, 0.2, 0.2)},
            {'id': 'wall_2', 'pos': Point(x=-5.0, y=0.0, z=1.0), 'size': [0.2, 8.0, 2.0], 'color': (0.8, 0.2, 0.2)},
            {'id': 'wall_3', 'pos': Point(x=0.0, y=5.0, z=1.0), 'size': [8.0, 0.2, 2.0], 'color': (0.8, 0.2, 0.2)},
            {'id': 'wall_4', 'pos': Point(x=0.0, y=-5.0, z=1.0), 'size': [8.0, 0.2, 2.0], 'color': (0.8, 0.2, 0.2)},
            {'id': 'table', 'pos': Point(x=2.0, y=2.0, z=0.4), 'size': [1.0, 0.8, 0.8], 'color': (0.6, 0.4, 0.2)},
            {'id': 'box_1', 'pos': Point(x=-2.0, y=-1.0, z=0.3), 'size': [0.6, 0.6, 0.6], 'color': (0.2, 0.6, 0.8)},
            {'id': 'cylinder_1', 'pos': Point(x=1.5, y=-2.5, z=0.75), 'size': [0.4, 1.5], 'color': (0.8, 0.6, 0.2)},
        ]

        for obs in obstacles:
            obj = EnvironmentObject(
                id=obs['id'],
                type='static',
                shape='box' if len(obs['size']) == 3 else 'cylinder',
                position=obs['pos'],
                orientation=Vector3(x=0.0, y=0.0, z=0.0),
                size=obs['size'],
                color=obs['color']
            )
            self.environment_objects[obs['id']] = obj

    def add_object_callback(self, msg):
        """Callback for adding objects to the environment"""
        # Parse object specification from message
        # Format: "type:shape:pos_x:pos_y:pos_z:size1:size2:size3:color_r:color_g:color_b"
        try:
            parts = msg.data.split(':')
            if len(parts) >= 8:
                obj_id = f"obj_{len(self.environment_objects)}"
                obj_type = parts[0]
                shape = parts[1]
                pos = Point(x=float(parts[2]), y=float(parts[3]), z=float(parts[4]))
                size = [float(parts[5]), float(parts[6]), float(parts[7])] if len(parts) >= 8 else [float(parts[5]), float(parts[6])]
                color = (float(parts[8]), float(parts[9]), float(parts[10])) if len(parts) >= 11 else (0.5, 0.5, 0.5)

                new_obj = EnvironmentObject(
                    id=obj_id,
                    type=obj_type,
                    shape=shape,
                    position=pos,
                    orientation=Vector3(x=0.0, y=0.0, z=0.0),
                    size=size,
                    color=color
                )
                self.environment_objects[obj_id] = new_obj
                self.get_logger().info(f'Added object {obj_id} to environment')
        except Exception as e:
            self.get_logger().error(f'Error parsing add object command: {e}')

    def remove_object_callback(self, msg):
        """Callback for removing objects from the environment"""
        obj_id = msg.data
        if obj_id in self.environment_objects:
            del self.environment_objects[obj_id]
            self.get_logger().info(f'Removed object {obj_id} from environment')
        else:
            self.get_logger().warn(f'Tried to remove non-existent object: {obj_id}')

    def environment_update_callback(self):
        """Main environment update callback"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update dynamic objects (if any)
        self.update_dynamic_objects(dt)

        # Publish environment visualization
        self.publish_environment_visualization()

        # Publish occupancy grid
        self.publish_occupancy_grid()

        # Publish static objects list
        self.publish_static_objects()

        # Publish environment status
        status_msg = String()
        status_msg.data = f"ENVIRONMENT: Objects={len(self.environment_objects)}, Time={current_time:.2f}"
        self.environment_status_publisher.publish(status_msg)

    def update_dynamic_objects(self, dt: float):
        """Update positions of dynamic objects"""
        for obj_id, obj in self.environment_objects.items():
            if obj.type == 'dynamic' and obj.dynamic_properties:
                # Apply simple motion based on dynamic properties
                if 'velocity' in obj.dynamic_properties:
                    vel = obj.dynamic_properties['velocity']
                    obj.position.x += vel[0] * dt
                    obj.position.y += vel[1] * dt
                    obj.position.z += vel[2] * dt

                # Boundary checking
                if obj.position.x < self.environment_bounds['min_x'] or obj.position.x > self.environment_bounds['max_x']:
                    if 'velocity' in obj.dynamic_properties:
                        obj.dynamic_properties['velocity'][0] *= -1  # Bounce
                        obj.position.x = max(self.environment_bounds['min_x'], min(self.environment_bounds['max_x'], obj.position.x))

                if obj.position.y < self.environment_bounds['min_y'] or obj.position.y > self.environment_bounds['max_y']:
                    if 'velocity' in obj.dynamic_properties:
                        obj.dynamic_properties['velocity'][1] *= -1  # Bounce
                        obj.position.y = max(self.environment_bounds['min_y'], min(self.environment_bounds['max_y'], obj.position.y))

    def publish_environment_visualization(self):
        """Publish environment as visualization markers"""
        marker_array = MarkerArray()
        marker_id = 0

        for obj_id, obj in self.environment_objects.items():
            marker = Marker()
            marker.header.frame_id = "sim_world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker_id += 1

            # Set position and orientation
            marker.pose.position = obj.position
            marker.pose.orientation.w = 1.0  # No rotation for simplicity
            marker.pose.orientation.x = obj.orientation.x
            marker.pose.orientation.y = obj.orientation.y
            marker.pose.orientation.z = obj.orientation.z

            # Set scale based on object size
            if obj.shape == 'box':
                marker.type = Marker.CUBE
                marker.scale.x = obj.size[0]
                marker.scale.y = obj.size[1]
                marker.scale.z = obj.size[2] if len(obj.size) > 2 else 1.0
            elif obj.shape == 'sphere':
                marker.type = Marker.SPHERE
                marker.scale.x = obj.size[0] * 2  # Diameter
                marker.scale.y = obj.size[0] * 2
                marker.scale.z = obj.size[0] * 2
            elif obj.shape == 'cylinder':
                marker.type = Marker.CYLINDER
                marker.scale.x = obj.size[0] * 2  # Diameter
                marker.scale.y = obj.size[0] * 2
                marker.scale.z = obj.size[1]  # Height
            else:
                marker.type = Marker.CUBE  # Default
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0

            # Set color
            marker.color.r = obj.color[0]
            marker.color.g = obj.color[1]
            marker.color.b = obj.color[2]
            marker.color.a = 0.8  # Slightly transparent

            marker.action = Marker.ADD
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()  # Refresh rate

            marker_array.markers.append(marker)

        self.environment_publisher.publish(marker_array)

    def publish_occupancy_grid(self):
        """Publish occupancy grid representation of the environment"""
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = "sim_map"
        occupancy_grid.info.resolution = self.grid_resolution
        occupancy_grid.info.width = self.grid_width
        occupancy_grid.info.height = self.grid_height
        occupancy_grid.info.origin.position.x = self.environment_bounds['min_x']
        occupancy_grid.info.origin.position.y = self.environment_bounds['min_y']
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0

        # Initialize grid with unknown (-1)
        grid_data = [-1] * (self.grid_width * self.grid_height)

        # Mark occupied cells based on environment objects
        for obj_id, obj in self.environment_objects.items():
            if obj.shape in ['box', 'cylinder'] and obj.type == 'static':
                # Calculate grid cells occupied by this object
                obj_min_x = obj.position.x - obj.size[0] / 2
                obj_max_x = obj.position.x + obj.size[0] / 2
                obj_min_y = obj.position.y - (obj.size[1] / 2 if len(obj.size) > 1 else obj.size[0] / 2)
                obj_max_y = obj.position.y + (obj.size[1] / 2 if len(obj.size) > 1 else obj.size[0] / 2)

                # Convert object bounds to grid coordinates
                grid_min_x = int((obj_min_x - self.environment_bounds['min_x']) / self.grid_resolution)
                grid_max_x = int((obj_max_x - self.environment_bounds['min_x']) / self.grid_resolution)
                grid_min_y = int((obj_min_y - self.environment_bounds['min_y']) / self.grid_resolution)
                grid_max_y = int((obj_max_y - self.environment_bounds['min_y']) / self.grid_resolution)

                # Clamp to grid bounds
                grid_min_x = max(0, min(self.grid_width - 1, grid_min_x))
                grid_max_x = max(0, min(self.grid_width - 1, grid_max_x))
                grid_min_y = max(0, min(self.grid_height - 1, grid_min_y))
                grid_max_y = max(0, min(self.grid_height - 1, grid_max_y))

                # Mark cells as occupied (100)
                for x in range(grid_min_x, grid_max_x + 1):
                    for y in range(grid_min_y, grid_max_y + 1):
                        grid_idx = y * self.grid_width + x
                        if 0 <= grid_idx < len(grid_data):
                            grid_data[grid_idx] = 100

        occupancy_grid.data = grid_data
        self.occupancy_grid_publisher.publish(occupancy_grid)

    def publish_static_objects(self):
        """Publish list of static objects"""
        static_objects = PoseArray()
        static_objects.header.stamp = self.get_clock().now().to_msg()
        static_objects.header.frame_id = "sim_world"

        for obj_id, obj in self.environment_objects.items():
            if obj.type == 'static':
                pose = Pose()
                pose.position = obj.position
                pose.orientation.w = 1.0
                static_objects.poses.append(pose)

        self.static_objects_publisher.publish(static_objects)

    def add_dynamic_object(self, obj_type: str, shape: str, position: Point, size: List[float],
                          velocity: List[float] = None, color: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        """Add a dynamic object to the environment"""
        obj_id = f"dynamic_{len([o for o in self.environment_objects.values() if o.type == 'dynamic'])}"

        dynamic_props = {'velocity': velocity if velocity else [0.0, 0.0, 0.0]} if velocity else {}

        new_obj = EnvironmentObject(
            id=obj_id,
            type='dynamic',
            shape=shape,
            position=position,
            orientation=Vector3(x=0.0, y=0.0, z=0.0),
            size=size,
            color=color,
            dynamic_properties=dynamic_props
        )
        self.environment_objects[obj_id] = new_obj
        self.get_logger().info(f'Added dynamic object {obj_id} at ({position.x}, {position.y}, {position.z})')

    def generate_random_environment(self, num_objects: int = 10):
        """Generate a random environment with specified number of objects"""
        self.environment_objects.clear()
        self.initialize_default_environment()  # Keep the ground plane

        for i in range(num_objects):
            obj_type = random.choice(['static', 'decoration'])
            shape = random.choice(['box', 'sphere', 'cylinder'])

            # Random position within bounds (avoiding center area for robot)
            pos_x = random.uniform(self.environment_bounds['min_x'] + 1, self.environment_bounds['max_x'] - 1)
            pos_y = random.uniform(self.environment_bounds['min_y'] + 1, self.environment_bounds['max_y'] - 1)
            pos_z = random.uniform(0.1, 1.0)  # Above ground

            # Random size
            if shape == 'box':
                size = [random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), random.uniform(0.2, 1.5)]
            elif shape == 'sphere':
                size = [random.uniform(0.1, 0.5)]  # radius
            else:  # cylinder
                size = [random.uniform(0.1, 0.4), random.uniform(0.2, 1.2)]  # radius, height

            color = (random.random(), random.random(), random.random())

            obj = EnvironmentObject(
                id=f'rand_obj_{i}',
                type=obj_type,
                shape=shape,
                position=Point(x=pos_x, y=pos_y, z=pos_z),
                orientation=Vector3(x=0.0, y=0.0, z=0.0),
                size=size,
                color=color
            )
            self.environment_objects[f'rand_obj_{i}'] = obj

        self.get_logger().info(f'Generated random environment with {num_objects} objects')

    def get_environment_objects(self) -> Dict[str, EnvironmentObject]:
        """Get a copy of all environment objects"""
        return self.environment_objects.copy()

    def get_occupancy_at(self, x: float, y: float) -> int:
        """Get occupancy value at specific coordinates"""
        grid_x = int((x - self.environment_bounds['min_x']) / self.grid_resolution)
        grid_y = int((y - self.environment_bounds['min_y']) / self.grid_resolution)

        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            grid_idx = grid_y * self.grid_width + grid_x
            if hasattr(self, '_last_grid_data') and 0 <= grid_idx < len(self._last_grid_data):
                return self._last_grid_data[grid_idx]

        return -1  # Unknown


def main(args=None):
    rclpy.init(args=args)

    environment_modeler_node = EnvironmentModelerNode()

    try:
        rclpy.spin(environment_modeler_node)
    except KeyboardInterrupt:
        pass
    finally:
        environment_modeler_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()