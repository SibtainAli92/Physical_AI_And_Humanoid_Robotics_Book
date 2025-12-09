from setuptools import find_packages, setup

package_name = 'ros2_nervous_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS 2 Core Nervous System for Humanoid Robotics',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'central_nervous_node = ros2_nervous_system.nodes.central_nervous_node:main',
            'sensor_fusion_node = ros2_nervous_system.nodes.sensor_fusion_node:main',
            'motor_control_node = ros2_nervous_system.nodes.motor_control_node:main',
            'behavior_manager_node = ros2_nervous_system.nodes.behavior_manager_node:main',
            'communication_hub_node = ros2_nervous_system.nodes.communication_hub_node:main',
        ],
    },
)