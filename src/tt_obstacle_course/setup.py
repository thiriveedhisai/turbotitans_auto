import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'tt_obstacle_course'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='u_manohs',
    maintainer_email='user@example.com',
    description='Autonomous obstacle-course stack for the Turbo Titans RC truck.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'{name} = {package_name}.{name}:main' for name in (
                'terrain_monitor', 'start_detector', 'marker_detector', 'course_perception',
                'behavior_executor', 'speed_governor', 'twist_to_ackermann', 'mission_manager')
        ],
    },
)
