"""Keyboard teleop that publishes ackermann_msgs/AckermannDrive.

Styled after turtlebot3_teleop_keyboard, but maps keys to a car-like command:
steering angle (rad) and speed (m/s) instead of a Twist.

Controls:
        w
   a    s    d
        x

  w / x : increase / decrease speed (forward +, reverse -)
  a / d : steer left / right
  s or space : full stop and center steering
  CTRL-C to quit
"""

import os
import select
import sys

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

MAX_SPEED = 2.0          # m/s
MAX_STEER = 0.5          # rad
SPEED_STEP = 0.1         # m/s per key press
STEER_STEP = 0.05        # rad per key press

MSG = """
Control Your RC Car!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease speed (~ +/- {speed_step} m/s)
a/d : steer left/right (~ +/- {steer_step} rad)
s / space : force stop and center steering

CTRL-C to quit
""".format(speed_step=SPEED_STEP, steer_step=STEER_STEP)


def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def constrain(value, low, high):
    return max(low, min(high, value))


def make_simple_profile(output, target, step):
    if target > output:
        output = min(target, output + step)
    elif target < output:
        output = max(target, output - step)
    else:
        output = target
    return output


def print_status(speed, steer):
    print(f'currently:\tspeed {speed:.2f} m/s\tsteer {steer:.2f} rad')


def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    node = rclpy.create_node('rc_car_teleop_keyboard')
    pub = node.create_publisher(AckermannDrive, 'ackermann_cmd', 10)

    target_speed = 0.0
    target_steer = 0.10
    control_speed = 0.0
    control_steer = 0.0

    try:
        print(MSG)
        while True:
            key = get_key(settings)
            if key == 'w':
                target_speed = constrain(target_speed + SPEED_STEP,
                                         -MAX_SPEED, MAX_SPEED)
                print_status(target_speed, target_steer)
            elif key == 'x':
                target_speed = constrain(target_speed - SPEED_STEP,
                                         -MAX_SPEED, MAX_SPEED)
                print_status(target_speed, target_steer)
            elif key == 'd':
                target_steer = constrain(target_steer + STEER_STEP,
                                         -MAX_STEER, MAX_STEER)
                print_status(target_speed, target_steer)
            elif key == 'a':
                target_steer = constrain(target_steer - STEER_STEP,
                                         -MAX_STEER, MAX_STEER)
                print_status(target_speed, target_steer)
            elif key == 'f':
                target_steer = 0.10
                print_status(target_speed, target_steer)

            elif key == ' ' or key == 's':
                target_speed = 0.0
                target_steer = 0.10
                control_speed = 0.0
                control_steer = 0.0
                print_status(target_speed, target_steer)
            else:
                if key == '\x03':  # CTRL-C
                    break

            control_speed = make_simple_profile(
                control_speed, target_speed, SPEED_STEP / 2.0)
            control_steer = make_simple_profile(
                control_steer, target_steer, STEER_STEP / 2.0)

            msg = AckermannDrive()
            msg.speed = float(control_speed)
            msg.steering_angle = float(control_steer)
            pub.publish(msg)

    except Exception as exc:
        print(exc)
    finally:
        # Publish a stop command on exit.
        msg = AckermannDrive()
        msg.speed = 0.0
        msg.steering_angle = 0.0
        pub.publish(msg)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
