"""2-D course simulator standing in for the truck, ZED and lidar.

Consumes the governor's /ackermann_cmd exactly like rc_car_controller would,
and publishes what the real drivers publish:
  /zed/zed_node/odom (+ TF odom -> zed_camera_link), with optional drift
  /zed/zed_node/imu/data  attitude from ramps/bank, vibration on gravel/potholes
  /scan                   ray-marched against the 2.5-D course (tilt hits floor)
  /zone_event             when a zone marker is within 1.5 m and in view
It also auto-starts the race, detects collisions and falls, scores the run and
writes results + a trajectory plot to the output directory.
"""

import json
import math
import os
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from tt_course_sim.course import FALL_DROP, Course

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class CourseSim(Node):

    def __init__(self):
        super().__init__('course_sim')
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.seed = int(p('seed', 0))
        self.course = Course(rails=bool(p('rails', True)), seed=self.seed)
        self.reached_car_wash = False
        self.true_radius = float(p('true_min_turn_radius', 1.52))
        self.max_steer = float(p('max_steer_cmd', 0.5))
        self.servo_tau = float(p('servo_tau', 0.08))
        self.speed_tau = float(p('speed_tau', 0.25))
        self.lidar_h = float(p('lidar_height', 0.25))
        self.laser_x = float(p('laser_offset_x', -0.15))
        self.rear_axle = float(p('rear_axle_behind_camera', 0.38))
        self.drift = float(p('odom_yaw_drift_deg_per_m', 0.3))
        self.scale_err = float(p('odom_scale_error', 0.01))
        self.start_delay = float(p('start_delay', 4.0))
        self.timeout = float(p('timeout', 600.0))
        self.markers_on = bool(p('markers', True))
        self.out_dir = p('output_dir', '/tmp/tt_sim')
        start_zone = str(p('start_zone', ''))
        self.rng = np.random.default_rng(self.seed + 1)

        x, y, yaw = self.course.waypoint(start_zone) if start_zone else self.course.start_pose
        self.x, self.y, self.yaw, self.z = x, y, yaw, 0.0
        self.v = 0.0
        self.steer = 0.0
        self.cmd = AckermannDrive()
        self.cmd_t = -1e9
        self.ox, self.oy, self.oyaw = 0.0, 0.0, 0.0     # odometry (drifts)
        self.nose_up = self.roll = 0.0
        self.vib = 0.0
        self.zone = 'start'
        self.t0 = time.monotonic()
        self.race_t = None
        self.done = False
        self.result = {}
        self.events = []
        self.collisions = 0
        self.last_collision = -1e9
        self.zone_log = {}
        self.mission_zone = None
        self.traj = []
        self.steps = 0
        self.markers_sent = set()
        self.max_speed_seen = 0.0
        self.last_progress = (0.0, x, y)

        self.tf = TransformBroadcaster(self)
        self.pub_odom = self.create_publisher(Odometry, '/zed/zed_node/odom', 10)
        self.pub_imu = self.create_publisher(Imu, '/zed/zed_node/imu/data', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)
        self.pub_zone = self.create_publisher(String, '/zone_event', 10)
        self.pub_start = self.create_publisher(Bool, '/race_start', LATCHED)
        self.pub_status = self.create_publisher(String, '/sim/status', 10)
        self.create_subscription(AckermannDrive, '/ackermann_cmd', self.on_cmd, 10)
        self.create_subscription(String, '/mission/state', self.on_mission, 10)
        self.dt = 0.01
        self.create_timer(self.dt, self.physics)
        self.create_timer(1 / 30.0, self.publish_odom)
        self.create_timer(0.1, self.publish_scan)
        self.create_timer(0.5, self.supervise)
        self.get_logger().info(f'Course ready (rails={self.course.rails}, seed={self.seed}), '
                               f'race starts in {self.start_delay:.0f} s')

    # ----------------------------------------------------------------- input
    def on_cmd(self, msg):
        self.cmd, self.cmd_t = msg, time.monotonic()

    def on_mission(self, msg):
        try:
            self.mission_zone = json.loads(msg.data).get('zone')
        except ValueError:
            pass

    def now(self):
        return time.monotonic() - self.t0

    def event(self, kind, text):
        t = self.race_time()
        self.events.append({'t': round(t, 2), 'kind': kind, 'zone': self.zone, 'text': text,
                            'x': round(self.x, 2), 'y': round(self.y, 2)})
        text = f'[{t:6.1f}s] {kind}: {text} (zone {self.zone})'
        # rclpy forbids changing severity at one call site, so branch explicitly.
        if kind in ('FALL', 'TIMEOUT', 'STUCK'):
            self.get_logger().error(text)
        elif kind == 'COLLISION':
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)

    def race_time(self):
        return 0.0 if self.race_t is None else self.now() - self.race_t

    # --------------------------------------------------------------- physics
    def footprint(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        us = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.11, -0.22, -0.33, -0.43, -0.43, -0.43,
                       -0.43, -0.43, -0.33, -0.22, -0.11])
        vs = np.array([0.1, 0.05, 0.0, -0.05, -0.1, -0.1, -0.1, -0.1, -0.1, -0.05, 0.0, 0.05,
                       0.1, 0.1, 0.1, 0.1]) * 1.015
        return x + c * us - s * vs, y + s * us + c * vs

    def physics(self):
        if self.done:
            return
        dt = self.dt
        if self.race_t is None and self.now() > self.start_delay:
            self.race_t = self.now()
            self.pub_start.publish(Bool(data=True))
            self.get_logger().info('RACE START published')
        fresh = time.monotonic() - self.cmd_t < 0.5
        target_v = self.cmd.speed if fresh else 0.0
        target_s = self.cmd.steering_angle if fresh else self.steer
        self.steer += (target_s - self.steer) * min(1.0, dt / self.servo_tau)
        # Gravity on slopes; the real controller's speed loop fights this.
        self.v += (target_v - self.v) * min(1.0, dt / self.speed_tau) \
            - 9.81 * math.sin(self.nose_up) * dt * 0.3
        frac = max(-1.0, min(1.0, self.steer / self.max_steer))
        kappa = frac / self.true_radius
        # Bicycle model about the rear axle; the pose we track is the camera.
        ra = self.rear_axle
        rx, ry = self.x - ra * math.cos(self.yaw), self.y - ra * math.sin(self.yaw)
        ds = self.v * dt
        nyaw = self.yaw + ds * kappa
        rx += ds * math.cos((self.yaw + nyaw) / 2)
        ry += ds * math.sin((self.yaw + nyaw) / 2)
        nx, ny = rx + ra * math.cos(nyaw), ry + ra * math.sin(nyaw)
        z, gx, gy, vib, zone = self.course.ground(nx, ny, self.z)

        if z - self.z > 0.06:
            blocked = True               # driving into the side of a ramp/bridge
        else:
            fx, fy = self.footprint(nx, ny, nyaw)
            blocked = self.course.blocked(fx, fy, z)
        if blocked:
            if self.now() - self.last_collision > 1.0:
                self.collisions += 1
                self.event('COLLISION', f'v={self.v:.2f} m/s')
            self.last_collision = self.now()
            self.v = 0.0
            return
        if self.z - z > FALL_DROP:
            self.event('FALL', f'dropped {self.z - z:.2f} m off the course')
            self.finish('fell')
            return

        # Odometry integrates the same motion with drift and scale error.
        dyaw = nyaw - self.yaw
        self.oyaw += dyaw + math.radians(self.drift) * abs(ds) * self.rng.normal()
        self.ox += ds * (1 + self.scale_err) * math.cos(self.oyaw)
        self.oy += ds * (1 + self.scale_err) * math.sin(self.oyaw)

        self.x, self.y, self.yaw, self.z = nx, ny, math.atan2(math.sin(nyaw), math.cos(nyaw)), z
        h = (math.cos(self.yaw), math.sin(self.yaw))
        self.nose_up = math.atan(gx * h[0] + gy * h[1])
        self.roll = math.atan(-gx * h[1] + gy * h[0])
        self.vib = vib * min(1.0, abs(self.v) / 0.5)
        if zone:
            if zone != self.zone and self.race_t is not None:
                self.zone_log[zone] = round(self.race_time(), 2)
            self.zone = zone
        self.max_speed_seen = max(self.max_speed_seen, abs(self.v))
        self.publish_imu(kappa)
        self.steps += 1
        if self.race_t is not None and self.steps % 5 == 0:
            self.traj.append((round(self.race_time(), 2), round(self.x, 3), round(self.y, 3),
                              round(self.v, 2), self.zone, self.mission_zone or ''))
        if self.zone == 'car_wash':
            self.reached_car_wash = True
        if self.reached_car_wash and self.course.in_box(self.course.finish, self.x, self.y):
            self.finish('finished')

    # -------------------------------------------------------------- sensors
    def stamp(self):
        return self.get_clock().now().to_msg()

    def publish_imu(self, kappa):
        m = Imu()
        m.header.stamp = self.stamp()
        m.header.frame_id = 'zed_imu_link'
        # REP-103: positive pitch is nose down.
        q = rpy_to_quat(self.roll, -self.nose_up, self.oyaw)
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = q
        m.angular_velocity.z = self.v * kappa
        noise = self.rng.normal(0.0, 0.05 + self.vib)
        m.linear_acceleration.z = 9.81 * math.cos(self.nose_up) * math.cos(self.roll) + noise
        m.linear_acceleration.x = 9.81 * math.sin(self.nose_up)
        self.pub_imu.publish(m)

    def publish_odom(self):
        stamp = self.stamp()
        q = rpy_to_quat(self.roll, -self.nose_up, self.oyaw)
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = 'odom'
        o.child_frame_id = 'zed_camera_link'
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = \
            self.ox, self.oy, self.z
        (o.pose.pose.orientation.x, o.pose.pose.orientation.y,
         o.pose.pose.orientation.z, o.pose.pose.orientation.w) = q
        o.twist.twist.linear.x = self.v
        self.pub_odom.publish(o)
        t = TransformStamped()
        t.header = o.header
        t.child_frame_id = 'zed_camera_link'
        t.transform.translation.x, t.transform.translation.y = self.ox, self.oy
        t.transform.translation.z = self.z
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = q
        self.tf.sendTransform(t)

    def publish_scan(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx, ly = self.x + c * self.laser_x, self.y + s * self.laser_x
        ranges = self.course.scan(lx, ly, self.z + self.lidar_h, self.yaw, self.nose_up, self.roll)
        for box in self.course.noise_boxes:
            if self.course.in_box(box, self.x, self.y) or self.course.in_box(box, *self.ahead(0.8)):
                a = -math.pi + np.arange(ranges.size) * (2 * math.pi / ranges.size)
                front = np.abs(a) < math.radians(70)
                ribbons = front & (self.rng.random(ranges.size) < 0.35)
                ranges = np.where(ribbons, self.rng.uniform(0.15, 1.2, ranges.size), ranges)
        m = LaserScan()
        m.header.stamp = self.stamp()
        m.header.frame_id = 'laser'
        m.angle_min = -math.pi
        m.angle_increment = 2 * math.pi / ranges.size
        m.angle_max = m.angle_min + m.angle_increment * (ranges.size - 1)
        m.range_min, m.range_max = 0.15, 8.0
        m.scan_time = 0.1
        m.ranges = [float(r) for r in ranges]
        self.pub_scan.publish(m)
        if self.markers_on:
            self.check_markers()

    def ahead(self, d):
        return self.x + d * math.cos(self.yaw), self.y + d * math.sin(self.yaw)

    def check_markers(self):
        for zone, mx, my in self.course.markers:
            dx, dy = mx - self.x, my - self.y
            bearing = math.atan2(dy, dx) - self.yaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if math.hypot(dx, dy) < 1.5 and abs(bearing) < math.radians(50):
                self.pub_zone.publish(String(data=zone))

    # ------------------------------------------------------------ supervisor
    def supervise(self):
        if self.done or self.race_t is None:
            return
        t = self.race_time()
        status = {'t': round(t, 1), 'zone': self.zone, 'mission_zone': self.mission_zone,
                  'v': round(self.v, 2), 'collisions': self.collisions}
        self.pub_status.publish(String(data=json.dumps(status)))
        progress_t, px, py = self.last_progress
        if math.hypot(self.x - px, self.y - py) > 0.3:
            self.last_progress = (t, self.x, self.y)
        elif t - progress_t > 45.0:
            self.event('STUCK', 'no progress for 45 s')
            self.finish('stuck')
        if t > self.timeout:
            self.event('TIMEOUT', f'{self.timeout:.0f} s')
            self.finish('timeout')

    def finish(self, outcome, exit_node=True):
        if self.done:
            return
        self.done = True
        self.v = 0.0
        t = self.race_time()
        order = ['start', 'ramp_up', 'bridge', 'helix', 'tunnel', 'narrow', 'gravel',
                 'bank_approach', 'bank_uturn', 'pothole', 'wide', 'buckets', 'hoops',
                 'car_wash']
        reached = [z for z in order if z in self.zone_log or z == 'start']
        self.result = {'outcome': outcome, 'time_s': round(t, 1), 'collisions': self.collisions,
                       'furthest_zone': reached[-1] if reached else 'start',
                       'zones_reached': self.zone_log, 'max_speed': round(self.max_speed_seen, 2),
                       'rails': self.course.rails, 'seed': self.seed, 'events': self.events}
        summary = (f'=== RUN {outcome.upper()} in {t:.1f} s, collisions={self.collisions}, '
                   f'furthest zone={self.result["furthest_zone"]} ===')
        if outcome == 'finished':
            self.get_logger().info(summary)
        else:
            self.get_logger().error(summary)
        self.write_outputs()
        if exit_node:
            raise SystemExit(0)

    def write_outputs(self):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, 'result.json'), 'w') as f:
            json.dump(self.result, f, indent=2)
        with open(os.path.join(self.out_dir, 'trajectory.csv'), 'w') as f:
            f.write('t,x,y,v,true_zone,mission_zone\n')
            for row in self.traj:
                f.write(','.join(str(v) for v in row) + '\n')
        try:
            from tt_course_sim.plot import plot_run
            plot_run(self.course, self.traj, self.events, self.result,
                     os.path.join(self.out_dir, 'run.png'))
        except Exception as exc:  # plotting must never hide the result
            self.get_logger().warn(f'plot failed: {exc}')
        self.get_logger().info(f'results in {self.out_dir}')


def main(args=None):
    rclpy.init(args=args)
    node = CourseSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if not node.done:
            node.finish('aborted', exit_node=False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
