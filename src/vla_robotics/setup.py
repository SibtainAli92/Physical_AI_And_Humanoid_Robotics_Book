from setuptools import find_packages, setup

package_name = 'vla_robotics'

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
    description='Vision-Language-Action Robotics for Humanoid Robotics',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_language_node = vla_robotics.nodes.vision_language_node:main',
            'action_planner_node = vla_robotics.nodes.action_planner_node:main',
            'multimodal_fusion_node = vla_robotics.nodes.multimodal_fusion_node:main',
            'language_understanding_node = vla_robotics.nodes.language_understanding_node:main',
            'task_execution_node = vla_robotics.nodes.task_execution_node:main',
        ],
    },
)