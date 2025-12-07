from setuptools import find_packages, setup

package_name = 'vla_capstone'

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
    description='Vision-Language-Action Capstone Project for Humanoid Robotics',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_listener_node = vla_capstone.nodes.voice_listener_node:main',
            'planning_agent_node = vla_capstone.nodes.planning_agent_node:main',
            'task_manager_node = vla_capstone.nodes.task_manager_node:main',
            'navigation_node = vla_capstone.nodes.navigation_node:main',
            'perception_node = vla_capstone.nodes.perception_node:main',
            'manipulation_node = vla_capstone.nodes.manipulation_node:main',
        ],
    },
)