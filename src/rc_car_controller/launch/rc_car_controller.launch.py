import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('rc_car_controller'),
        'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='rc_car_controller',
            executable='rc_car_controller_node',
            name='rc_car_controller',
            output='screen',
            parameters=[params],
        ),
    ])
