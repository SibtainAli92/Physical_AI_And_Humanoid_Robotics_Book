#!/usr/bin/env python3
# simulation_setup.py
# Setup script for Isaac Sim digital twin environment

import omni
from omni.isaac.kit import SimulationApp
import numpy as np
import carb

# Simulation configuration
CONFIG = {
    'headless': False,
    'physics_dt': 1.0/1000.0,  # 1ms physics timestep
    'rendering_dt': 1.0/60.0,  # 60Hz rendering
    'stage_units_in_meters': 1.0
}

# Start Isaac Sim application
simulation_app = SimulationApp(CONFIG)

# Import Isaac Sim modules after starting the app
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.semantics import add_semantic_classification

# Create world instance
world = World(stage_units_in_meters=1.0)

def setup_humanoid_robot(urdf_path: str, prim_path: str = "/World/HumanoidRobot"):
    """
    Setup the humanoid robot in the simulation
    """
    print(f"Adding humanoid robot from URDF: {urdf_path}")

    # Add the humanoid robot from URDF
    add_reference_to_stage(
        usd_path=urdf_path,
        prim_path=prim_path
    )

    # Create robot instance
    robot = Articulation(prim_path=prim_path)

    return robot

def setup_environment():
    """
    Setup the simulation environment with basic objects
    """
    print("Setting up simulation environment...")

    # Add ground plane
    from omni.isaac.core.utils.prims import create_primitive
    create_primitive(
        prim_path="/World/GroundPlane",
        prim_type="Plane",
        scale=np.array([10.0, 10.0, 1.0]),
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )

    # Add a simple obstacle
    create_primitive(
        prim_path="/World/Box",
        prim_type="Cube",
        scale=np.array([0.5, 0.5, 0.5]),
        position=np.array([2.0, 0.0, 0.25]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )

    # Set camera view
    set_camera_view(eye=np.array([3, 3, 3]), target=np.array([0, 0, 1]))

def setup_sensors(robot_path: str):
    """
    Setup sensors for the robot
    """
    print("Setting up sensors...")

    # Add IMU sensor to the head
    from omni.isaac.sensor import IMUSensor
    imu_sensor = IMUSensor(
        prim_path=f"{robot_path}/head/Imu_Sensor",
        frequency=100
    )

    return imu_sensor

def main():
    """
    Main function to setup and run the simulation
    """
    print("Starting Humanoid Robot Simulation Setup...")

    # Get assets root path
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets path")
        return

    # Setup environment
    setup_environment()

    # For this example, we'll use a temporary URDF path
    # In a real scenario, this would point to your humanoid robot URDF
    robot_urdf_path = "/path/to/humanoid_robot.urdf"  # This will be replaced with actual path

    # Try to add a simple robot for demonstration
    try:
        # First, let's try to create a simple articulated robot
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        assets_root_path = get_assets_root_path()

        if assets_root_path:
            # Add a sample robot if available
            robot_path = f"{assets_root_path}/Isaac/Robots/Franka/franka.usd"
            add_reference_to_stage(
                usd_path=robot_path,
                prim_path="/World/Robot"
            )
        else:
            # If no assets are available, create a simple cube as placeholder
            from omni.isaac.core.utils.prims import create_primitive
            create_primitive(
                prim_path="/World/Robot",
                prim_type="Cube",
                scale=np.array([0.3, 0.3, 0.3]),
                position=np.array([0.0, 0.0, 0.5])
            )
    except Exception as e:
        print(f"Could not add robot: {e}")
        # Create a simple cube as a placeholder
        from omni.isaac.core.utils.prims import create_primitive
        create_primitive(
            prim_path="/World/Robot",
            prim_type="Cube",
            scale=np.array([0.3, 0.3, 0.3]),
            position=np.array([0.0, 0.0, 0.5])
        )

    # Set camera view
    set_camera_view(eye=np.array([2, 2, 2]), target=np.array([0, 0, 1]))

    # Reset the world
    world.reset()

    # Run simulation for a few steps to verify setup
    print("Running simulation for verification...")
    for i in range(100):
        world.step(render=True)
        if i % 50 == 0:
            print(f"Simulation step {i}")

    print("Simulation setup completed successfully!")

    # Keep the simulation running
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()