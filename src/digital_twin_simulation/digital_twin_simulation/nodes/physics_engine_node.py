#!/usr/bin/env python3

"""
Physics Engine Node for Humanoid Robot Digital Twin

This node implements the physics simulation engine for the digital twin,
handling collision detection, rigid body dynamics, and environmental physics.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3, Point, Wrench
from geometry_msgs.msg import Twist
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class RigidBody:
    """Represents a rigid body in the physics simulation"""
    name: str
    mass: float
    position: Point
    orientation: Vector3
    linear_velocity: Vector3
    angular_velocity: Vector3
    inertia_tensor: List[float]  # 3x3 inertia tensor as 9-element list
    collision_shape: str  # 'sphere', 'box', 'capsule', etc.
    size: List[float]  # Dimensions of the shape


class PhysicsEngineNode(Node):
    def __init__(self):
        super().__init__('physics_engine_node')

        # Publishers for physics data
        self.physics_state_publisher = self.create_publisher(JointState, 'physics/joint_states', 10)
        self.collision_publisher = self.create_publisher(String, 'physics/collisions', 10)
        self.contact_forces_publisher = self.create_publisher(Wrench, 'physics/contact_forces', 10)
        self.physics_status_publisher = self.create_publisher(String, 'physics/status', 10)

        # Subscribers for simulation inputs
        self.sim_joint_state_subscriber = self.create_subscription(
            JointState, 'sim/joint_states', self.sim_joint_state_callback, 10)
        self.external_force_subscriber = self.create_subscription(
            Wrench, 'physics/external_forces', self.external_force_callback, 10)

        # Timer for physics simulation
        self.physics_timer = self.create_timer(0.001, self.physics_update_callback)  # 1000 Hz for accurate physics

        # Physics state
        self.rigid_bodies: Dict[str, RigidBody] = {}
        self.gravity = Vector3(x=0.0, y=0.0, z=-9.81)
        self.sim_time = time.time()
        self.last_update_time = time.time()
        self.physics_enabled = True
        self.collision_detection_enabled = True

        # Initialize humanoid robot rigid bodies
        self.initialize_robot_bodies()

        # Physics parameters
        self.linear_damping = 0.99
        self.angular_damping = 0.99
        self.collision_margin = 0.001  # 1mm collision margin
        self.max_penetration_depth = 0.01  # 1cm maximum penetration

        self.get_logger().info('Physics Engine Node initialized')

    def initialize_robot_bodies(self):
        """Initialize rigid bodies for the humanoid robot"""
        # Define basic humanoid body parts
        robot_parts = [
            {'name': 'torso', 'mass': 10.0, 'position': Point(x=0, y=0, z=0.8), 'size': [0.3, 0.2, 0.5], 'shape': 'box'},
            {'name': 'head', 'mass': 2.0, 'position': Point(x=0, y=0, z=1.1), 'size': [0.2, 0.2, 0.2], 'shape': 'sphere'},
            {'name': 'upper_arm_left', 'mass': 1.5, 'position': Point(x=-0.2, y=0, z=0.7), 'size': [0.08, 0.3, 0.08], 'shape': 'capsule'},
            {'name': 'upper_arm_right', 'mass': 1.5, 'position': Point(x=0.2, y=0, z=0.7), 'size': [0.08, 0.3, 0.08], 'shape': 'capsule'},
            {'name': 'lower_arm_left', 'mass': 1.0, 'position': Point(x=-0.4, y=0, z=0.7), 'size': [0.06, 0.25, 0.06], 'shape': 'capsule'},
            {'name': 'lower_arm_right', 'mass': 1.0, 'position': Point(x=0.4, y=0, z=0.7), 'size': [0.06, 0.25, 0.06], 'shape': 'capsule'},
            {'name': 'upper_leg_left', 'mass': 2.0, 'position': Point(x=-0.1, y=0, z=0.4), 'size': [0.09, 0.4, 0.09], 'shape': 'capsule'},
            {'name': 'upper_leg_right', 'mass': 2.0, 'position': Point(x=0.1, y=0, z=0.4), 'size': [0.09, 0.4, 0.09], 'shape': 'capsule'},
            {'name': 'lower_leg_left', 'mass': 1.5, 'position': Point(x=-0.1, y=0, z=0.1), 'size': [0.08, 0.35, 0.08], 'shape': 'capsule'},
            {'name': 'lower_leg_right', 'mass': 1.5, 'position': Point(x=0.1, y=0, z=0.1), 'size': [0.08, 0.35, 0.08], 'shape': 'capsule'},
            {'name': 'foot_left', 'mass': 0.8, 'position': Point(x=-0.1, y=0, z=-0.1), 'size': [0.15, 0.08, 0.25], 'shape': 'box'},
            {'name': 'foot_right', 'mass': 0.8, 'position': Point(x=0.1, y=0, z=-0.1), 'size': [0.15, 0.08, 0.25], 'shape': 'box'},
        ]

        for part in robot_parts:
            body = RigidBody(
                name=part['name'],
                mass=part['mass'],
                position=part['position'],
                orientation=Vector3(x=0.0, y=0.0, z=0.0),
                linear_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                inertia_tensor=self.calculate_inertia_tensor(part['shape'], part['size'], part['mass']),
                collision_shape=part['shape'],
                size=part['size']
            )
            self.rigid_bodies[part['name']] = body

    def calculate_inertia_tensor(self, shape: str, size: List[float], mass: float) -> List[float]:
        """Calculate the inertia tensor for a given shape and size"""
        # Simplified inertia tensor calculation
        if shape == 'sphere':
            # I = (2/5) * m * r^2 for all axes
            r = size[0]  # assuming radius is first dimension
            I = (2/5) * mass * r * r
            return [I, 0, 0, 0, I, 0, 0, 0, I]
        elif shape == 'box':
            # Ixx = (1/12) * m * (h^2 + d^2)
            # Iyy = (1/12) * m * (w^2 + d^2)
            # Izz = (1/12) * m * (w^2 + h^2)
            w, h, d = size[0], size[1], size[2]
            Ixx = (1/12) * mass * (h*h + d*d)
            Iyy = (1/12) * mass * (w*w + d*d)
            Izz = (1/12) * mass * (w*w + h*h)
            return [Ixx, 0, 0, 0, Iyy, 0, 0, 0, Izz]
        elif shape == 'capsule':
            # Simplified capsule inertia (approximated as cylinder)
            r, h = size[0], size[1]
            I_perp = (1/4) * mass * r*r + (1/12) * mass * h*h  # perpendicular to length
            I_para = (1/2) * mass * r*r  # parallel to length
            return [I_para, 0, 0, 0, I_perp, 0, 0, 0, I_perp]
        else:
            # Default for unknown shapes
            return [mass*0.1, 0, 0, 0, mass*0.1, 0, 0, 0, mass*0.1]

    def sim_joint_state_callback(self, msg):
        """Callback for simulated joint state updates"""
        # Update rigid body positions based on joint angles
        # This is a simplified approach - in reality would use forward kinematics
        for i, joint_name in enumerate(msg.name):
            if i < len(msg.position):
                # Map joint angles to body part positions (simplified)
                self.update_body_from_joint(joint_name, msg.position[i])

    def update_body_from_joint(self, joint_name: str, joint_angle: float):
        """Update body position based on joint angle"""
        # This is a simplified implementation
        # In a real system, this would use forward kinematics
        pass

    def external_force_callback(self, msg):
        """Callback for external forces applied to the system"""
        # Apply external forces to rigid bodies
        # For now, we'll apply forces to the torso as an example
        if 'torso' in self.rigid_bodies:
            torso = self.rigid_bodies['torso']
            # Apply force to linear velocity
            dt = 0.001  # Fixed time step for physics
            torso.linear_velocity.x += (msg.force.x / torso.mass) * dt
            torso.linear_velocity.y += (msg.force.y / torso.mass) * dt
            torso.linear_velocity.z += (msg.force.z / torso.mass) * dt

            # Apply torque to angular velocity
            torso.angular_velocity.x += (msg.torque.x / (torso.mass * 0.1)) * dt  # Simplified moment of inertia
            torso.angular_velocity.y += (msg.torque.y / (torso.mass * 0.1)) * dt
            torso.angular_velocity.z += (msg.torque.z / (torso.mass * 0.1)) * dt

    def physics_update_callback(self):
        """Main physics update callback"""
        if not self.physics_enabled:
            return

        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update all rigid bodies
        for body_name, body in self.rigid_bodies.items():
            self.update_rigid_body(body, dt)

        # Perform collision detection
        if self.collision_detection_enabled:
            self.detect_collisions()

        # Publish physics state
        self.publish_physics_state()

        # Publish physics status
        status_msg = String()
        status_msg.data = f"PHYSICS_RUNNING: Bodies={len(self.rigid_bodies)}, dt={dt:.6f}s"
        self.physics_status_publisher.publish(status_msg)

    def update_rigid_body(self, body: RigidBody, dt: float):
        """Update a single rigid body based on physics"""
        # Apply gravity
        body.linear_velocity.z += self.gravity.z * dt

        # Apply damping
        body.linear_velocity.x *= self.linear_damping
        body.linear_velocity.y *= self.linear_damping
        body.linear_velocity.z *= self.linear_damping

        body.angular_velocity.x *= self.angular_damping
        body.angular_velocity.y *= self.angular_damping
        body.angular_velocity.z *= self.angular_damping

        # Update position based on velocity
        body.position.x += body.linear_velocity.x * dt
        body.position.y += body.linear_velocity.y * dt
        body.position.z += body.linear_velocity.z * dt

        # Update orientation based on angular velocity
        body.orientation.x += body.angular_velocity.x * dt
        body.orientation.y += body.angular_velocity.y * dt
        body.orientation.z += body.angular_velocity.z * dt

        # Simple ground collision (z=0 plane)
        body_size_z = body.size[2] if len(body.size) > 2 else 0.1
        ground_level = 0.0  # Ground at z=0

        if body.position.z - body_size_z/2 < ground_level:
            body.position.z = ground_level + body_size_z/2
            body.linear_velocity.z = -body.linear_velocity.z * 0.3  # Bounce with damping
            body.linear_velocity.x *= 0.8  # Friction
            body.linear_velocity.y *= 0.8

    def detect_collisions(self):
        """Detect collisions between rigid bodies"""
        collision_occurred = False
        collision_pairs = []

        body_list = list(self.rigid_bodies.values())

        # Simple collision detection between all pairs
        for i in range(len(body_list)):
            for j in range(i + 1, len(body_list)):
                body1 = body_list[i]
                body2 = body_list[j]

                # Simple sphere collision detection (approximating all shapes as spheres)
                distance = math.sqrt(
                    (body1.position.x - body2.position.x)**2 +
                    (body1.position.y - body2.position.y)**2 +
                    (body1.position.z - body2.position.z)**2
                )

                # Approximate collision distance as sum of "radii"
                radius1 = max(body1.size) / 2
                radius2 = max(body2.size) / 2
                collision_distance = radius1 + radius2

                if distance < collision_distance + self.collision_margin:
                    collision_occurred = True
                    collision_pairs.append((body1.name, body2.name))

                    # Simple collision response (impulse-based)
                    self.resolve_collision(body1, body2)

        # Publish collision information
        if collision_occurred:
            collision_msg = String()
            collision_msg.data = f"COLLISION: {', '.join([f'{pair[0]}-{pair[1]}' for pair in collision_pairs])}"
            self.collision_publisher.publish(collision_msg)

    def resolve_collision(self, body1: RigidBody, body2: RigidBody):
        """Resolve collision between two bodies"""
        # Calculate collision normal (from body1 to body2)
        dx = body2.position.x - body1.position.x
        dy = body2.position.y - body1.position.y
        dz = body2.position.z - body1.position.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance == 0:  # Avoid division by zero
            return

        # Normalize collision normal
        nx = dx / distance
        ny = dy / distance
        nz = dz / distance

        # Relative velocity
        rel_vel_x = body2.linear_velocity.x - body1.linear_velocity.x
        rel_vel_y = body2.linear_velocity.y - body1.linear_velocity.y
        rel_vel_z = body2.linear_velocity.z - body1.linear_velocity.z

        # Velocity along normal
        vel_along_normal = rel_vel_x * nx + rel_vel_y * ny + rel_vel_z * nz

        # Don't resolve if velocities are separating
        if vel_along_normal > 0:
            return

        # Calculate impulse scalar
        e = 0.3  # Coefficient of restitution
        j = -(1 + e) * vel_along_normal
        j /= (1/body1.mass + 1/body2.mass)

        # Apply impulse
        impulse_x = j * nx
        impulse_y = j * ny
        impulse_z = j * nz

        body1.linear_velocity.x -= impulse_x / body1.mass
        body1.linear_velocity.y -= impulse_y / body1.mass
        body1.linear_velocity.z -= impulse_z / body1.mass

        body2.linear_velocity.x += impulse_x / body2.mass
        body2.linear_velocity.y += impulse_y / body2.mass
        body2.linear_velocity.z += impulse_z / body2.mass

    def publish_physics_state(self):
        """Publish physics state to ROS topics"""
        # Publish joint states (simulated from rigid body positions)
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = 'physics_world'

        # Map rigid body positions to joint-like representation
        for body_name, body in self.rigid_bodies.items():
            joint_state_msg.name.append(f"{body_name}_joint")
            joint_state_msg.position.append(math.sqrt(body.position.x**2 + body.position.y**2 + body.position.z**2))
            joint_state_msg.velocity.append(math.sqrt(body.linear_velocity.x**2 + body.linear_velocity.y**2 + body.linear_velocity.z**2))
            joint_state_msg.effort.append(0.0)  # Simplified

        self.physics_state_publisher.publish(joint_state_msg)

    def enable_physics(self, enable: bool):
        """Enable or disable physics simulation"""
        self.physics_enabled = enable
        self.get_logger().info(f"Physics simulation {'enabled' if enable else 'disabled'}")

    def enable_collision_detection(self, enable: bool):
        """Enable or disable collision detection"""
        self.collision_detection_enabled = enable
        self.get_logger().info(f"Collision detection {'enabled' if enable else 'disabled'}")


def main(args=None):
    rclpy.init(args=args)

    physics_engine_node = PhysicsEngineNode()

    try:
        rclpy.spin(physics_engine_node)
    except KeyboardInterrupt:
        # Disable physics before shutdown to prevent unexpected behavior
        physics_engine_node.enable_physics(False)
        pass
    finally:
        physics_engine_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()