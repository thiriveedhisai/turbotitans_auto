from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Keyboard teleop needs an interactive terminal for stdin, so run it in a
    # new xterm. Requires xterm (sudo apt install xterm). To run without xterm,
    # use:  ros2 run rc_car_controller teleop_keyboard
    return LaunchDescription([
        Node(
            package='rc_car_controller',
            executable='teleop_keyboard',
            name='rc_car_teleop_keyboard',
            output='screen',
            prefix='xterm -e',
            emulate_tty=True,
        ),
    ])
