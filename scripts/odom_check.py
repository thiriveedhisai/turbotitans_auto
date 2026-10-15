#!/usr/bin/env python3
"""Standalone odometry check (no tt_obstacle_course imports): distance scale and heading
error of the ZED odometry, from two taped legs (default 5 m) with a known turn (90 deg)
between them. Same maths as the tt_dash "Odometry check" panel.

Needs only the ZED driver running (or the whole stack), e.g.
  ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
  python3 scripts/odom_check.py                         # 5 m, 90 deg, raw ZED odometry
  python3 scripts/odom_check.py --leg1 4.0 --leg2 3.5 --turn 90
  python3 scripts/odom_check.py --odom /odom/local --current-scale 1.03   # stack's scaled odometry

Move the truck by hand or by teleop; pushing it slowly along the tape is the most exact.
At each of the 4 points press Enter (r + Enter = redo the previous mark, q = quit):
  1 START of tape 1, lined up on it      2 END of tape 1
  3 START of tape 2, lined up on it      4 END of tape 2
Prints scale = true / odometry distance and the value to put in tuning.yaml
(odom_guard: odom_scale), plus turn and drift errors of the ZED heading and the gyro.
Saves everything to --out (default ~/runs/odocheck_<time>.json).
"""

import argparse
import json
import math
import os
import threading
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

STEPS = ['put the truck at the START of tape 1, lined up with it',
         'move it slowly along tape 1 to its END and stop',
         'turn onto tape 2 (any way, a K-turn is fine) and line up at its START',
         'move it slowly along tape 2 to its END and stop']


def wrapd(a):
    """Radians -> degrees in (-180, 180]."""
    return math.degrees(math.atan2(math.sin(a), math.cos(a)))


class Listener(Node):

    def __init__(self, odom_topic, imu_topic):
        super().__init__('odom_check')
        self.lock = threading.Lock()
        self.odom = None
        self.odom_t = None
        self.gyro = 0.0
        self.imu_t = None
        self.imu_seen = False
        self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        self.create_subscription(Imu, imu_topic, self.on_imu, qos_profile_sensor_data)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.odom, self.odom_t = (p.x, p.y, yaw), time.monotonic()

    def on_imu(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            if self.imu_t is not None and 0.0 < t - self.imu_t < 0.5:
                self.gyro += m.angular_velocity.z * (t - self.imu_t)
            self.imu_t, self.imu_seen = t, True

    def snapshot(self):
        with self.lock:
            fresh = self.odom is not None and time.monotonic() - self.odom_t < 1.0
            return (*self.odom, self.gyro) if fresh else None


def result(marks, true_m, true_deg, cur_scale, raw_zed):
    (x0, y0, v0, g0), (x1, y1, v1, g1), (x2, y2, v2, g2), (x3, y3, v3, g3) = marks
    d1, d2 = math.hypot(x1 - x0, y1 - y0), math.hypot(x3 - x2, y3 - y2)
    s1 = true_m[0] / d1 if d1 > 0.5 else None
    s2 = true_m[1] / d2 if d2 > 0.5 else None
    good = [s for s in (s1, s2) if s is not None]
    scale = sum(good) / len(good) if good else None
    path_turn = wrapd(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y0, x1 - x0))
    true_turn = math.copysign(true_deg, path_turn)
    # odom_guard multiplies the RAW ZED odometry by odom_scale: measured on the raw topic the
    # scale is the value itself; measured on its (already scaled) output, multiply.
    suggested = None if scale is None else (scale if raw_zed else cur_scale * scale)
    return {
        'leg1_true_m': true_m[0], 'leg2_true_m': true_m[1], 'true_turn_deg': true_turn,
        'leg1_odom_m': round(d1, 3), 'leg2_odom_m': round(d2, 3),
        'scale_leg1': s1 and round(s1, 4), 'scale_leg2': s2 and round(s2, 4),
        'scale': scale and round(scale, 4), 'suggested_odom_scale': suggested and round(suggested, 4),
        'path_turn_err_deg': round(path_turn - true_turn, 2),
        'zed_turn_err_deg': round(wrapd(math.radians(wrapd(v2 - v0) - true_turn)), 2),
        'gyro_turn_err_deg': round(math.degrees(g2 - g0) - true_turn, 2),
        'zed_drift_leg1_deg': round(wrapd(v1 - v0), 2), 'zed_drift_leg2_deg': round(wrapd(v3 - v2), 2),
        'gyro_drift_leg1_deg': round(math.degrees(g1 - g0), 2),
        'gyro_drift_leg2_deg': round(math.degrees(g3 - g2), 2),
        'warning': None if len(good) == 2 else 'a leg was under 0.5 m by odometry: redo the check',
    }


def report(r, imu_seen):
    def f(v, unit=''):
        return '-' if v is None else f'{v}{unit}'
    print('\n================ odometry check ================')
    print(f'leg 1: true {r["leg1_true_m"]:.2f} m, odometry {r["leg1_odom_m"]:.3f} m -> scale {f(r["scale_leg1"])}')
    print(f'leg 2: true {r["leg2_true_m"]:.2f} m, odometry {r["leg2_odom_m"]:.3f} m -> scale {f(r["scale_leg2"])}')
    if r['scale'] is not None:
        off = (r['scale'] - 1.0) * 100
        print(f'SCALE {r["scale"]:.4f}  (odometry reads {-off:+.1f}% vs the tape)')
        print(f'  -> tuning.yaml  odom_guard: odom_scale: {r["suggested_odom_scale"]}'
              + ('   (within 2%: fine to leave as is)' if abs(off) < 2 else ''))
    print(f'turn (true {r["true_turn_deg"]:+.0f} deg) error: path {r["path_turn_err_deg"]:+.1f}, '
          f'ZED heading {r["zed_turn_err_deg"]:+.1f}'
          + (f', gyro {r["gyro_turn_err_deg"]:+.1f} deg' if imu_seen else ' deg (no IMU data)'))
    print(f'heading drift on the straight legs (should be ~0): ZED {r["zed_drift_leg1_deg"]:+.1f} / '
          f'{r["zed_drift_leg2_deg"]:+.1f} deg'
          + (f', gyro {r["gyro_drift_leg1_deg"]:+.1f} / {r["gyro_drift_leg2_deg"]:+.1f} deg' if imu_seen else ''))
    print('  turn error over ~3 deg or drift over ~2 deg per 5 m: heading needs attention')
    if r['warning']:
        print('WARNING:', r['warning'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    ap.add_argument('--leg1', type=float, default=5.0, help='true length of tape 1 (m)')
    ap.add_argument('--leg2', type=float, default=5.0, help='true length of tape 2 (m)')
    ap.add_argument('--turn', type=float, default=90.0, help='true angle between the tapes (deg)')
    ap.add_argument('--odom', default='/zed/zed_node/odom', help='odometry topic to check')
    ap.add_argument('--imu', default='/zed/zed_node/imu/data')
    ap.add_argument('--current-scale', type=float, default=1.0,
                    help='odom_scale in use (only matters with --odom /odom/local)')
    ap.add_argument('--out', default='~/runs')
    a, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = Listener(a.odom, a.imu)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    raw_zed = a.odom.startswith('/zed')
    print(f'odometry: {a.odom}   IMU: {a.imu}   tapes {a.leg1:.2f} / {a.leg2:.2f} m, turn {a.turn:.0f} deg')
    print('waiting for odometry ...', end='', flush=True)
    while node.snapshot() is None:
        time.sleep(0.2)
    print(' ok\n')

    marks = []
    try:
        while len(marks) < 4:
            k = input(f'[{len(marks) + 1}/4] {STEPS[len(marks)]}, then Enter: ').strip().lower()
            if k == 'q':
                return
            if k == 'r':
                if marks:
                    marks.pop()
                    print('  previous mark removed')
                continue
            s = node.snapshot()
            if s is None:
                print('  no fresh odometry (ZED running?), mark not taken')
                continue
            marks.append(s)
            msg = f'  mark {len(marks)}: x={s[0]:+.3f} y={s[1]:+.3f} yaw={math.degrees(s[2]):+.1f} deg'
            if len(marks) in (2, 4):
                p = marks[-2]
                msg += f'   leg {len(marks) // 2} by odometry: {math.hypot(s[0] - p[0], s[1] - p[1]):.3f} m'
            print(msg)
        r = result(marks, (a.leg1, a.leg2), a.turn, a.current_scale, raw_zed)
        report(r, node.imu_seen)
        out = os.path.expanduser(a.out)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, time.strftime('odocheck_%m%d_%H%M%S.json'))
        with open(path, 'w') as fh:
            json.dump(dict(r, odom_topic=a.odom, marks=marks, odom_scale_in_use=a.current_scale),
                      fh, indent=1)
        print(f'saved {path}')
    except (KeyboardInterrupt, EOFError, ExternalShutdownException):
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
