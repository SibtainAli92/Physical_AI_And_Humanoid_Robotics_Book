from setuptools import find_packages, setup

package_name = 'humanoid_testing'

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
    description='Testing framework for Humanoid Robotics Platform',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_runner = humanoid_testing.test_runner:main',
            'module_tester = humanoid_testing.module_tester:main',
            'integration_tester = humanoid_testing.integration_tester:main',
            'performance_tester = humanoid_testing.performance_tester:main',
        ],
    },
)