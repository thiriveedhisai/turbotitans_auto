#!/usr/bin/env python3
"""Standalone throttle calibration (no tt_obstacle_course imports): drive straight at a
few commanded speeds, measure the real speed with the ZED odometry, and print what to
put in tuning.yaml (rc_car_controller: speed_scale, or a measured fwd_speed_mps/fwd_pulse_us
table).

Run ONLY the motor node and the ZED (not the obstacle-course stack: its speed governor
also publishes /ackermann_cmd):
  ros2 launch rc_car_controller rc_car_controller.launch.py
  ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
  python3 scripts/speed_calib.py                       # 0.3 0.5 0.8 1.0 m/s, 2 m each
  python3 scripts/speed_calib.py --speeds 0.25 0.4 0.6 --distance 3

Needs a straight clear run of --distance + 1 m per speed. Each run stops on its own after
--distance metres by odometry or --time seconds; Ctrl-C stops the truck at once.
The pulse column assumes rc_car_controller's linear map (speed_map_mode linear):
  pulse = neutral + v / max_speed * (max_us - neutral)   [defaults 1500, 2.0 m/s, 2000]
"""

import argparse
import math
import threading
import time

import rclpy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rclpy.node import Node


class Calib(Node):

    def __init__(self, cmd_topic, odom_topic):
        super().__init__('speed_calib')
        self.lock = threading.Lock()
        self.track = []
        self.odom_t = None
        self.pub = self.create_publisher(AckermannDrive, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 50)

    def on_odom(self, m):
        p = m.pose.pose.position
        with self.lock:
            self.odom_t = time.monotonic()
            self.track.append((self.odom_t, p.x, p.y))
            if len(self.track) > 5000:
                del self.track[:1000]

    def since(self, t0):
        with self.lock:
            return [s for s in self.track if s[0] >= t0]

    def odom_ok(self):
        with self.lock:
            return self.odom_t is not None and time.monotonic() - self.odom_t < 1.0

    def send(self, v):
        self.pub.publish(AckermannDrive(speed=float(v), steering_angle=0.0))

    def stop(self, secs=0.5):
        end = time.monotonic() + secs
        while time.monotonic() < end:
            self.send(0.0)
            time.sleep(0.05)


def path_length(pts):
    return sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(pts, pts[1:]))


def run_one(node, v, distance, max_time):
    """Drive at v until `distance` m by odometry or max_time s; real speed from the last
    60% of the run (after the truck has got up to speed)."""
    t0 = time.monotonic()
    while True:
        node.send(v)
        time.sleep(0.05)
        pts = node.since(t0)
        dt = time.monotonic() - t0
        if path_length(pts) >= distance or dt >= max_time:
            break
    node.stop()
    pts = node.since(t0)
    total = path_length(pts)
    if len(pts) < 5 or total < 0.05:
        return 0.0, total, dt
    t_split = pts[0][0] + 0.4 * (pts[-1][0] - pts[0][0])
    steady = [s for s in pts if s[0] >= t_split]
    span = steady[-1][0] - steady[0][0]
    return (path_length(steady) / span if span > 0.2 else 0.0), total, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    ap.add_argument('--speeds', type=float, nargs='+', default=[0.3, 0.5, 0.8, 1.0],
                    help='commanded speeds to try (m/s, forward)')
    ap.add_argument('--distance', type=float, default=2.0, help='stop each run after this (m)')
    ap.add_argument('--time', type=float, default=6.0, help='or after this (s)')
    ap.add_argument('--cmd', default='/ackermann_cmd')
    ap.add_argument('--odom', default='/zed/zed_node/odom')
    ap.add_argument('--neutral', type=float, default=1500.0)
    ap.add_argument('--max-us', type=float, default=2000.0)
    ap.add_argument('--max-speed', type=float, default=2.0, help="rc_car_controller max_speed")
    a, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = Calib(a.cmd, a.odom)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    print('waiting for odometry ...', end='', flush=True)
    while not node.odom_ok():
        time.sleep(0.2)
    print(' ok')

    def pulse(v):
        return a.neutral + min(v, a.max_speed) / a.max_speed * (a.max_us - a.neutral)

    rows = []
    try:
        for v in a.speeds:
            k = input(f'\n{v:.2f} m/s (pulse {pulse(v):.0f} us): truck straight with '
                      f'{a.distance + 1:.1f} m clear ahead, Enter = go, s = skip, q = quit: ')
            if k.strip().lower() == 'q':
                break
            if k.strip().lower() == 's':
                continue
            real, total, dt = run_one(node, v, a.distance, a.time)
            rows.append((v, pulse(v), real))
            print(f'  drove {total:.2f} m in {dt:.1f} s; steady speed {real:.3f} m/s '
                  f'({real / v:.0%} of commanded)' if real > 0 else
                  f'  did not move ({total:.2f} m): pulse below the ESC deadband?')
    except (KeyboardInterrupt, EOFError):
        print('\nstopped')
    finally:
        node.stop(1.0)

    if rows:
        print('\n commanded   pulse    measured')
        for v, p, real in rows:
            print(f'  {v:5.2f} m/s  {p:5.0f} us  {real:6.3f} m/s')
        moving = [(v, p, r) for v, p, r in rows if r > 0.05]
        if moving:
            ratio = sum(r / v for v, _, r in moving) / len(moving)
            print(f'\nreal / commanded on average: {ratio:.2f}')
            print(f'Quick fix (linear map): tuning.yaml rc_car_controller: speed_scale: '
                  f'{1.0 / ratio:.3f}')
            pts = sorted({round(r, 3): p for _, p, r in moving}.items())
            print('Or an exact table (rc_car_controller in tuning.yaml):')
            print('    speed_map_mode: "table"')
            print(f'    fwd_speed_mps: [0.0, {", ".join(f"{r:.3f}" for r, _ in pts)}]')
            print(f'    fwd_pulse_us: [{a.neutral:.0f}, {", ".join(f"{p:.0f}" for _, p in pts)}]')
        stalled = [v for v, _, r in rows if r <= 0.05]
        if stalled:
            print(f'Did not move at {stalled} m/s: anything the stack asks below '
                  f'{max(stalled):.2f} m/s will stall the truck.')
    # Stop the spin thread before destroying the node (else rclpy aborts at exit).
    rclpy.try_shutdown()
    spin.join(2.0)
    node.destroy_node()


if __name__ == '__main__':
    main()
