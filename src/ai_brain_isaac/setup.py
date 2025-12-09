from setuptools import find_packages, setup

package_name = 'ai_brain_isaac'

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
    description='AI Brain (NVIDIA Isaac) for Humanoid Robotics',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = ai_brain_isaac.nodes.perception_node:main',
            'decision_maker_node = ai_brain_isaac.nodes.decision_maker_node:main',
            'learning_node = ai_brain_isaac.nodes.learning_node:main',
            'memory_node = ai_brain_isaac.nodes.memory_node:main',
            'cognitive_controller_node = ai_brain_isaac.nodes.cognitive_controller_node:main',
        ],
    },
)