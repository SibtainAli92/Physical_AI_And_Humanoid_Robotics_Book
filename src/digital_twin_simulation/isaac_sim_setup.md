# Isaac Sim Setup Guide

## Installation Requirements

### System Requirements
- NVIDIA GPU with RTX or GTX 1080+ (Compute Capability 6.0+)
- CUDA 11.8 or later
- 16GB+ RAM (32GB recommended)
- 50GB+ free disk space
- Windows 10/11 64-bit or Ubuntu 20.04/22.04

### Isaac Sim Installation

#### Option 1: Omniverse Launcher (Recommended)
1. Download NVIDIA Omniverse Launcher from https://www.nvidia.com/en-us/omniverse/
2. Install and run the launcher
3. Search for Isaac Sim and install
4. Launch Isaac Sim from the launcher

#### Option 2: Direct Download
1. Go to https://developer.nvidia.com/isaac-sim
2. Download the appropriate version for your OS
3. Extract to a directory with sufficient space

### Isaac Sim Configuration for Humanoid Robot

#### Robot Model Setup
Create a URDF/SDF model of the humanoid robot with appropriate physical properties:

```bash
# Isaac Sim project structure
isaac_sim_project/
├── assets/
│   ├── robots/
│   │   └── humanoid/
│   │       ├── model.urdf
│   │       ├── config.yaml
│   │       └── meshes/
│   │           ├── body.stl
│   │           ├── head.stl
│   │           ├── arm.stl
│   │           └── leg.stl
│   ├── environments/
│   │   ├── office/
│   │   ├── home/
│   │   └── outdoor/
│   └── materials/
└── scripts/
    ├── robot_spawn.py
    └── control_interface.py
```

#### Physics Properties Configuration
Configure mass, inertia, and friction properties in the URDF:

```xml
<!-- In your URDF file -->
<link name="humanoid_link">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
  <visual>
    <geometry>
      <mesh filename="meshes/body.stl"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <mesh filename="meshes/body.stl"/>
    </geometry>
  </collision>
</link>
```

### Isaac Sim Python API Setup

#### Environment Setup Script
```python
# setup_isaac_sim.py
import omni
from omni.isaac.kit import SimulationApp

# Configure simulation parameters
config = {
    'headless': False,
    'physics_dt': 1.0/60.0,
    'rendering_dt': 1.0/60.0,
    'stage_units_in_meters': 1.0
}

# Start Isaac Sim application
simulation_app = SimulationApp(config)

# Import Isaac Sim modules
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view

# Create world instance
world = World(stage_units_in_meters=1.0)

# Add humanoid robot to the stage
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets path")

# Add your humanoid robot
add_reference_to_stage(
    usd_path=f"{assets_root_path}/Isaac/Robots/Franka/franka.usd",
    prim_path="/World/Robot"
)

# Set camera view
set_camera_view(eye=[2, 2, 2], target=[0, 0, 1])

# Reset and step the world
world.reset()
for i in range(100):
    world.step(render=True)

# Shutdown simulation
simulation_app.close()
```

#### Robot Control Interface
```python
# robot_control.py
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

class HumanoidRobot(Robot):
    def __init__(self, prim_path: str, name: str = None, usd_path: str = None):
        super().__init__(
            prim_path=prim_path,
            name=name,
            usd_path=usd_path,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

    def set_joint_positions(self, positions: np.ndarray):
        """Set joint positions for the humanoid robot"""
        self.get_articulation_controller().apply_articulation_actions(
            ArticulationAction(joint_positions=positions)
        )

    def set_joint_velocities(self, velocities: np.ndarray):
        """Set joint velocities for the humanoid robot"""
        self.get_articulation_controller().apply_articulation_actions(
            ArticulationAction(joint_velocities=velocities)
        )

    def set_joint_efforts(self, efforts: np.ndarray):
        """Set joint efforts for the humanoid robot"""
        self.get_articulation_controller().apply_articulation_actions(
            ArticulationAction(joint_efforts=efforts)
        )
```

### Isaac Sim ROS Bridge Setup

To enable communication between Isaac Sim and ROS 2:

```bash
# Install Isaac ROS Bridge
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
cd isaac_ros_common
git checkout <appropriate_branch>

# Build the bridge
colcon build --symlink-install --packages-select \
    isaac_ros_common \
    isaac_ros_image_pipeline \
    isaac_ros_visual_slam
```

### Isaac Sim Launch Script

```bash
#!/bin/bash
# launch_isaac_sim.sh

# Set Isaac Sim path (adjust to your installation)
ISAAC_SIM_PATH="/path/to/isaac-sim"

# Launch Isaac Sim with specific configuration
${ISAAC_SIM_PATH}/python.sh -m omni.isaac.kit --enable-omni.kit.window.viewport --exec "your_simulation_script.py"

# Or use the Omniverse launcher directly
# ${ISAAC_SIM_PATH}/isaac-sim.exe
```

### Testing Isaac Sim Setup

Create a simple test to verify Isaac Sim is working:

```python
# test_isaac_sim.py
import omni
from omni.isaac.kit import SimulationApp

# Start simulation
config = {"headless": False}
simulation_app = SimulationApp(config)

# Import required modules
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import carb

# Create world
world = World(stage_units_in_meters=1.0)

# Add a simple object to test
add_reference_to_stage(
    usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Props/YCB/Axis_AL6061.usd",
    prim_path="/World/Axis"
)

# Run simulation for a few steps
world.reset()
for i in range(100):
    world.step(render=True)

# Close simulation
simulation_app.close()
print("Isaac Sim test completed successfully!")
```