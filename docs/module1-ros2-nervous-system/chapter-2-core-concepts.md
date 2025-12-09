---
sidebar_position: 3
---

# Chapter 2: Core Concepts of ROS 2 Nervous System

## Node Architecture

In ROS 2, nodes represent individual processes that perform specific functions within the robot system. Each node operates independently and communicates with other nodes through the ROS 2 communication infrastructure.

### Node Lifecycle
- **Creation**: Nodes are instantiated and initialized with specific parameters
- **Configuration**: Nodes receive configuration data and establish connections
- **Activation**: Nodes begin processing and communication
- **Deactivation**: Nodes can be paused while maintaining connections
- **Cleanup**: Nodes properly shut down and release resources

## Topics and Message Passing

Topics are the primary communication mechanism in ROS 2, enabling one-to-many communication through publish-subscribe patterns.

### Key Characteristics:
- **Publishers**: Nodes that send data to a topic
- **Subscribers**: Nodes that receive data from a topic
- **Message types**: Strongly typed data structures that define the format of information exchanged
- **Quality of Service (QoS)**: Configurable policies for reliability, durability, and performance

### Practical Robotics Example:
In a humanoid robot's walking system:
- Joint state publishers broadcast current position, velocity, and effort data
- Balance controllers subscribe to IMU data and joint states to maintain stability
- Visualization tools subscribe to various topics for monitoring and debugging

## Services and Actions

### Services
Services provide request-response communication for specific tasks that have a clear beginning and end:
- **Synchronous**: The caller waits for a response
- **One-to-one**: Single client communicates with single server
- **Use cases**: Calibration, configuration, and specific robot commands

### Actions
Actions are designed for long-running tasks that require feedback and goal management:
- **Goal**: Request to perform a specific task
- **Feedback**: Continuous updates during execution
- **Result**: Final outcome of the action
- **Use cases**: Navigation, manipulation, and complex robot behaviors

## Parameter Management

Parameters in ROS 2 allow for dynamic configuration of nodes:
- **Declaration**: Parameters must be declared before use
- **Types**: Support for various data types (integers, floats, strings, booleans, lists)
- **Management**: Parameters can be set at launch time or changed during runtime
- **Hierarchical**: Parameter names follow a namespace convention

## Quality of Service (QoS) Policies

QoS policies allow fine-tuning of communication behavior:
- **Reliability**: Reliable (all messages delivered) vs Best Effort (no guarantee)
- **Durability**: Volatile (new subscribers don't receive old messages) vs Transient Local (new subscribers receive last message)
- **History**: Keep All vs Keep Last (number of messages to store)
- **Deadline**: Maximum time between consecutive messages
- **Lifespan**: Maximum age of messages before they're dropped

## Communication Middleware

ROS 2 uses DDS (Data Distribution Service) as its default middleware:
- **Discovery**: Automatic detection of nodes and topics
- **Transport**: Handles message delivery across networks
- **Security**: Authentication, encryption, and access control
- **Performance**: Optimized for real-time and high-throughput applications