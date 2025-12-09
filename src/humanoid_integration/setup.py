from setuptools import find_packages, setup

package_name = 'humanoid_integration'

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
    description='Integration package for Humanoid Robotics System',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'system_integration_node = humanoid_integration.nodes.system_integration_node:main',
            'module_coordinator_node = humanoid_integration.nodes.module_coordinator_node:main',
            'safety_manager_node = humanoid_integration.nodes.safety_manager_node:main',
            'behavior_coordinator_node = humanoid_integration.nodes.behavior_coordinator_node:main',
        ],
    },
)