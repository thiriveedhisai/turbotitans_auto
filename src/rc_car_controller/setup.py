import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'rc_car_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='u_manohs',
    maintainer_email='user@example.com',
    description='AckermannDrive to PCA9685 PWM controller for an RC car.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rc_car_controller_node = '
            'rc_car_controller.rc_car_controller_node:main',
            'teleop_keyboard = '
            'rc_car_controller.teleop_keyboard:main',
        ],
    },
)
