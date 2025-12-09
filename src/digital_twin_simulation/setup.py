from setuptools import find_packages, setup

package_name = 'digital_twin_simulation'

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
    description='Digital Twin Simulation Environment for Humanoid Robotics',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulation_environment_node = digital_twin_simulation.nodes.simulation_environment_node:main',
            'physics_engine_node = digital_twin_simulation.nodes.physics_engine_node:main',
            'sensor_simulator_node = digital_twin_simulation.nodes.sensor_simulator_node:main',
            'environment_modeler_node = digital_twin_simulation.nodes.environment_modeler_node:main',
            'synchronization_node = digital_twin_simulation.nodes.synchronization_node:main',
        ],
    },
)