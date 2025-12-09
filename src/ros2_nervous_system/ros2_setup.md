# ROS 2 Humble Setup for Humanoid Robotics

## Installation Requirements

### System Requirements
- Ubuntu 22.04 LTS (recommended) or Windows 10/11 WSL2
- At least 8GB RAM (16GB recommended)
- 50GB+ free disk space
- Real-time capable kernel (for deterministic control)

### ROS 2 Humble Hawksbill Installation

#### Ubuntu 22.04 Installation
```bash
# Setup locale
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8

# Setup sources
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-moveit
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Real-time Kernel Configuration

For real-time performance, configure the kernel with PREEMPT_RT patches:

```bash
# Check current kernel version
uname -r

# Install real-time kernel (if available in repository)
sudo apt install linux-image-rt-generic

# Or compile from source (advanced users)
# Follow instructions at: https://wiki.linuxfoundation.org/realtime/start
```

### QoS Configuration

Create Quality of Service profiles for real-time communication:

```yaml
# config/qos_profiles.yaml
robot_state_publisher:
  ros__parameters:
    use_sim_time: false
    qos_overrides:
      /joint_states:
        publisher:
          depth: 10
          reliability: reliable
          durability: volatile
          history: keep_last
```

### Environment Setup

Create a setup script for ROS 2 workspace:

```bash
#!/bin/bash
# setup_ros2.sh

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f ~/humanoid_ws/install/setup.bash ]; then
    source ~/humanoid_ws/install/setup.bash
fi

# Set real-time parameters
ulimit -r 99  # Set real-time priority limit