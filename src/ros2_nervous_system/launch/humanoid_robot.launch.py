# humanoid_robot.launch.py
# ROS 2 launch file for coordinated node startup

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch.actions import Shutdown

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_description_path = LaunchConfiguration('robot_description_path',
        default=PathJoinSubstitution([FindPackageShare('humanoid_robot_description'), 'urdf', 'humanoid_robot.urdf.xacro']))

    # Set global parameters
    set_use_sim_time = SetParameter(name='use_sim_time', value=use_sim_time)

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description_path,
            {'use_sim_time': use_sim_time},
            {'publish_frequency': 50.0}
        ]
    )

    # Joint state broadcaster
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Humanoid controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            robot_description_path,
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Perception system node
    perception_node = Node(
        package='humanoid_perception',
        executable='perception_node',
        name='perception_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'camera_topic': '/camera/rgb/image_raw'},
            {'lidar_topic': '/lidar/points'}
        ],
        output='screen'
    )

    # Navigation system
    navigation_node = Node(
        package='nav2_bringup',
        executable='nav2_launch.py',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        condition=IfCondition(LaunchConfiguration('use_navigation', default='true'))
    )

    # AI brain interface
    ai_brain_interface = Node(
        package='ai_brain_isaac',
        executable='ai_brain_interface',
        name='ai_brain_interface',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Safety system monitor
    safety_monitor = Node(
        package='humanoid_safety',
        executable='safety_monitor',
        name='safety_monitor',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'emergency_stop_timeout': 0.1}
        ],
        output='screen'
    )

    # Return the launch description
    return LaunchDescription([
        set_use_sim_time,
        robot_state_publisher,
        controller_manager,
        joint_state_broadcaster,
        perception_node,
        ai_brain_interface,
        safety_monitor,
        navigation_node if LaunchConfiguration('use_navigation', default='true') else [],
    ])