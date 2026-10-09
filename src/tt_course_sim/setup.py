import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'tt_course_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='u_manohs',
    maintainer_email='user@example.com',
    description='2.5-D obstacle-course simulator for tt_obstacle_course.',
    license='Apache-2.0',
    entry_points={'console_scripts': ['course_sim = tt_course_sim.course_sim:main']},
)
