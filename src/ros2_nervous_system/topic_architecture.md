# Topic Architecture for Humanoid Robot System

## Overview
This document defines the topic architecture for inter-module communication in the humanoid robot system. The architecture follows ROS 2 best practices and ensures efficient, reliable communication between the various modules.

## Topic Naming Convention
- Use descriptive, lowercase names with underscores
- Group related topics under common namespaces
- Use consistent naming patterns across modules
- Follow ROS 2 REP-2011 conventions where applicable

## Core Communication Topics

### 1. Joint State Topics
```
/joint_states                    # Standard joint states (sensor_msgs/JointState)
/desired_joint_states            # Desired joint states (sensor_msgs/JointState)
/safe_joint_commands             # Safety-filtered joint commands (trajectory_msgs/JointTrajectory)
```

### 2. Sensor Data Topics
```
/imu/data                        # IMU sensor data (sensor_msgs/Imu)
/imu/data_raw                    # Raw IMU data (sensor_msgs/Imu)
/force_torque/left_foot          # Left foot force/torque (geometry_msgs/WrenchStamped)
/force_torque/right_foot         # Right foot force/torque (geometry_msgs/WrenchStamped)
/force_torque/left_hand          # Left hand force/torque (geometry_msgs/WrenchStamped)
/force_torque/right_hand         # Right hand force/torque (geometry_msgs/WrenchStamped)
/camera/rgb/image_raw            # RGB camera image (sensor_msgs/Image)
/camera/depth/image_raw          # Depth camera image (sensor_msgs/Image)
/lidar/points                    # LiDAR point cloud (sensor_msgs/PointCloud2)
```

### 3. Control Command Topics
```
/trajectory/joint_trajectory     # Joint trajectory commands (trajectory_msgs/JointTrajectory)
/trajectory/cartesian_pose       # Cartesian pose commands (geometry_msgs/PoseStamped)
/velocity_commands               # Velocity commands (std_msgs/Float64MultiArray)
/effort_commands                 # Effort commands (std_msgs/Float64MultiArray)
```

### 4. Perception Topics
```
/perception/objects              # Detected objects (object_msgs/ObjectArray)
/perception/human_poses          # Detected human poses (geometry_msgs/PoseArray)
/perception/occupancy_grid       # Occupancy grid map (nav_msgs/OccupancyGrid)
/perception/semantic_segmentation # Semantic segmentation (sensor_msgs/Image)
```

### 5. Navigation Topics
```
/navigation/goal_pose            # Navigation goal (geometry_msgs/PoseStamped)
/navigation/current_pose         # Current robot pose (geometry_msgs/PoseStamped)
/navigation/path                 # Planned path (nav_msgs/Path)
/navigation/cmd_vel              # Velocity commands for navigation (geometry_msgs/Twist)
```

### 6. AI/Brain Interface Topics
```
/ai/behavior_request             # Behavior requests from AI (std_msgs/String)
/ai/behavior_response            # Behavior responses to AI (std_msgs/String)
/ai/action_commands              # Action commands from AI (action_msgs/GoalInfo)
/ai/plan                         # High-level plans (nav_msgs/Path)
```

### 7. Safety System Topics
```
/emergency_stop                  # Emergency stop signal (std_msgs/Bool)
/joint_limit_violation          # Joint limit violations (std_msgs/Bool)
/collision_detected             # Collision detection (std_msgs/Bool)
/safety_status                  # Overall safety status (std_msgs/String)
```

## Quality of Service (QoS) Profiles

### 1. Sensor Data (High Frequency)
```yaml
/joint_states:
  reliability: reliable
  durability: volatile
  history: keep_last
  depth: 10

/imu/data:
  reliability: reliable
  durability: volatile
  history: keep_last
  depth: 100
  deadline: 0.01s  # 10ms deadline for real-time requirements
```

### 2. Control Commands (Real-time Critical)
```yaml
/trajectory/joint_trajectory:
  reliability: reliable
  durability: volatile
  history: keep_last
  depth: 1
  lifespan: 0.1s  # 100ms lifespan for old commands

/velocity_commands:
  reliability: reliable
  durability: volatile
  history: keep_last
  depth: 1
  deadline: 0.005s  # 5ms deadline for real-time control
```

### 3. Low Frequency Status
```yaml
/safety_status:
  reliability: reliable
  durability: transient_local
  history: keep_last
  depth: 1

/navigation/goal_pose:
  reliability: reliable
  durability: volatile
  history: keep_last
  depth: 1
```

## Module-Specific Topic Namespaces

### 1. ROS 2 Nervous System
```
/ros2_nervous_system/status      # System status
/ros2_nervous_system/health      # Health monitoring
```

### 2. Digital Twin Simulation
```
/simulation/ground_truth         # Ground truth from simulation (nav_msgs/Odometry)
/simulation/reset                # Reset simulation (std_msgs/Empty)
/simulation/pause                # Pause simulation (std_msgs/Bool)
```

### 3. AI Brain (Isaac)
```
/ai_brain/status                 # AI brain status
/ai_brain/intent                 # Human intent recognition
/ai_brain/emotion                # Emotion recognition
```

### 4. VLA Robotics
```
/vla/text_command               # Text commands (std_msgs/String)
/vla/visual_command             # Visual commands (sensor_msgs/Image)
/vla/action_execution           # Action execution status (std_msgs/String)
```

## Communication Patterns

### 1. Publisher-Subscriber Pattern
- Sensors publish data, multiple modules subscribe
- Control commands published by planners, executed by controllers
- Status information published by modules, monitored by health system

### 2. Service-Client Pattern
- Configuration requests
- Calibration services
- System management commands

### 3. Action-Based Pattern
- Long-running tasks (navigation, manipulation)
- Goal-based execution with feedback
- Cancelable operations

## Data Flow Examples

### 1. Walking Controller Data Flow
```
Perception Module -> /perception/occupancy_grid -> Navigation Module
Navigation Module -> /navigation/path -> Walking Controller
Walking Controller -> /trajectory/joint_trajectory -> Controller Manager
Controller Manager -> /joint_states -> Robot State Publisher
Robot State Publisher -> /tf -> All Modules
```

### 2. Safety System Data Flow
```
IMU -> /imu/data -> Safety Monitor
Force/Torque -> /force_torque/* -> Safety Monitor
Joint States -> /joint_states -> Safety Monitor
Safety Monitor -> /emergency_stop -> All Controllers
```

## Best Practices

### 1. Topic Design
- Use appropriate message types for the data being transmitted
- Consider bandwidth and frequency requirements
- Implement data compression for high-bandwidth topics
- Use latching for static data that new subscribers need

### 2. Performance Considerations
- Minimize the number of publishers per topic
- Use appropriate QoS settings for real-time requirements
- Consider message size and frequency trade-offs
- Implement topic throttling for high-frequency data

### 3. Security Considerations
- Use ROS 2 security features for sensitive topics
- Implement authentication for critical control topics
- Monitor topic access for unusual patterns
- Encrypt sensitive data transmission

## Topic Management

### 1. Lifecycle Management
- Properly initialize and cleanup topic publishers/subscribers
- Handle node lifecycle transitions appropriately
- Implement graceful degradation when topics are unavailable

### 2. Monitoring and Diagnostics
- Monitor topic publishing rates
- Check for message staleness
- Log communication errors
- Implement heartbeat mechanisms for critical topics

This topic architecture provides a robust foundation for inter-module communication in the humanoid robot system, ensuring efficient, reliable, and safe operation across all modules.