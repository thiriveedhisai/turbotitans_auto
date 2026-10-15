"""Arrow-key teleop in a terminal (needs a real terminal: run it with ros2 run, not in a launch).

  ros2 run tt_obstacle_course tt_teleop [--max 1.0]

Up/Down: speed +/-0.1 m/s (negative = reverse)   Left/Right: steer   Space: stop and centre
q / Esc: quit (commands stop; the truck stops 0.3 s later).
Publishes /teleop_cmd at 10 Hz: the speed governor gives it priority, e-stop still applies.
"""

import argparse
import curses
import time

import rclpy
from ackermann_msgs.msg import AckermannDrive


def loop(scr, node, pub, vmax, smax):
    scr.nodelay(True)
    curses.curs_set(0)
    speed = steer = 0.0
    while rclpy.ok():
        k = scr.getch()
        if k in (ord('q'), 27):
            break
        if k == curses.KEY_UP:
            speed = min(speed + 0.1, vmax)
        elif k == curses.KEY_DOWN:
            speed = max(speed - 0.1, -vmax)
        elif k == curses.KEY_LEFT:
            steer = min(steer + 0.1, smax)
        elif k == curses.KEY_RIGHT:
            steer = max(steer - 0.1, -smax)
        elif k == ord(' '):
            speed = steer = 0.0
        pub.publish(AckermannDrive(speed=float(speed), steering_angle=float(steer)))
        scr.erase()
        scr.addstr(0, 0, __doc__.split('\n\n')[1].split('\n')[0])
        scr.addstr(2, 0, f'speed {speed:+.1f} m/s   steer {steer:+.1f} rad (+ = left)')
        scr.addstr(3, 0, 'Up/Down speed, Left/Right steer, Space stop, q quit')
        scr.refresh()
        time.sleep(0.1)
    pub.publish(AckermannDrive())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--max', type=float, default=1.0, help='max speed (m/s)')
    ap.add_argument('--max-steer', type=float, default=0.5, help='max steering (rad)')
    a, ros_args = ap.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = rclpy.create_node('tt_teleop')
    pub = node.create_publisher(AckermannDrive, '/teleop_cmd', 10)
    try:
        curses.wrapper(loop, node, pub, a.max, a.max_steer)
    except KeyboardInterrupt:
        pub.publish(AckermannDrive())
    finally:
        node.destroy_node()
        rclpy.shutdown()
