"""Referee for the 3-D (Gazebo) run.

- Removes the red start cover at start_delay; the stack's camera start_detector
  must see the green light and publish /race_start (fallback after 10 s, logged).
- Tracks ground-truth pose (odometry publisher, odom frame = spawn pose).
- Ends the run on finish line, fall, flip, stuck or timeout; writes
  result.json, trajectory.csv and run.png to output_dir.
"""

import json
import math
import os
import subprocess

import rclpy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from tt_course_sim.course import Course

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
VOLATILE = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)


def quat_rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


class GzReferee(Node):

    def __init__(self):
        super().__init__('gz_referee')
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.seed = int(p('seed', 0))
        self.course = Course(rails=bool(p('rails', True)), seed=self.seed)
        self.start_zone = str(p('start_zone', ''))
        self.spawn = [float(v) for v in p('spawn', [0.0, 0.0, 0.0, 0.0])]
        self.out_dir = p('output_dir', '/tmp/tt_gz_run')
        self.timeout = float(p('timeout', 600.0))
        self.start_delay = float(p('start_delay', 6.0))
        self.world = p('world_name', 'course')

        self.pose = None
        self.t_start = None
        self.cover_removed = None
        self.detector_fired = None
        self.done = False
        self.traj = []
        self.events = []
        self.zones = {}
        self.zone = ''
        self.mission_zone = None
        self.markers_seen = []
        self.last_progress = None
        self.max_speed = 0.0

        self.pub_start = self.create_publisher(Bool, '/race_start', LATCHED)
        self.create_subscription(Odometry, '/zed/zed_node/odom', self.on_odom, 10)
        self.create_subscription(Bool, '/race_start', self.on_start, VOLATILE)
        self.create_subscription(String, '/mission/state', self.on_mission, 10)
        self.create_subscription(String, '/zone_event', self.on_marker, 10)
        self.cmd = (0.0, 0.0)
        self.create_subscription(AckermannDrive, '/ackermann_cmd', self.on_cmd, 10)
        self.create_timer(0.5, self.tick)
        self.t0 = None

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def race_time(self):
        return 0.0 if self.t_start is None else self.now() - self.t_start

    def event(self, kind, text):
        x, y = (self.pose[0], self.pose[1]) if self.pose else (0.0, 0.0)
        self.events.append({'t': round(self.race_time(), 2), 'kind': kind, 'zone': self.zone,
                            'text': text, 'x': round(x, 2), 'y': round(y, 2)})
        msg = f'[{self.race_time():6.1f}s] {kind}: {text} (zone {self.zone})'
        if kind in ('FALL', 'FLIP', 'STUCK', 'TIMEOUT', 'START_DETECTOR_FAILED'):
            self.get_logger().error(msg)
        else:
            self.get_logger().info(msg)

    # ------------------------------------------------------------- inputs
    def on_odom(self, msg):
        sx, sy, sz, syaw = self.spawn
        p = msg.pose.pose.position
        c, s = math.cos(syaw), math.sin(syaw)
        x, y, z = sx + c * p.x - s * p.y, sy + s * p.x + c * p.y, sz + p.z
        roll, pitch, yaw = quat_rpy(msg.pose.pose.orientation)
        yaw += syaw
        v = msg.twist.twist.linear.x
        self.pose = (x, y, z, roll, pitch, v)
        self.max_speed = max(self.max_speed, abs(v))
        zone = self.course.ground(x, y, z)[4]
        self.check_finish_line(x, y)
        if zone and zone != self.zone:
            if self.t_start is not None:
                self.zones.setdefault(zone, round(self.race_time(), 2))
            self.zone = zone
        if self.t_start is not None and (not self.traj or self.race_time() - self.traj[-1][0] >= 0.05):
            self.traj.append((round(self.race_time(), 2), round(x, 3), round(y, 3), round(v, 2),
                              self.zone, self.mission_zone or '',
                              round(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))), 1),
                              *self.cmd))

    def on_cmd(self, msg):
        self.cmd = (round(msg.speed, 2), round(msg.steering_angle, 3))

    def check_finish_line(self, x, y):
        """Finish = crossing the line (box centre, across its heading) after the car wash."""
        if self.done or self.t_start is None or 'car_wash' not in self.zones:
            return
        cx, cy, yaw, _, hw = self.course.finish
        u = (x - cx) * math.cos(yaw) + (y - cy) * math.sin(yaw)
        v = -(x - cx) * math.sin(yaw) + (y - cy) * math.cos(yaw)
        prev, self.finish_u = getattr(self, 'finish_u', None), u
        if prev is not None and prev < 0.0 <= u and abs(v) <= hw:
            self.finish('finished', 'FINISH', f'{self.race_time():.1f} s')

    def on_start(self, msg):
        if msg.data and self.detector_fired is None and self.cover_removed is not None:
            self.detector_fired = self.now()
            self.event('START', f'start detector fired {self.detector_fired - self.cover_removed:.2f} s '
                                f'after the light turned green')
        if msg.data and self.t_start is None:
            self.t_start = self.now()
            self.last_progress = (0.0, self.pose[0], self.pose[1]) if self.pose else None

    def on_mission(self, msg):
        try:
            self.mission_zone = json.loads(msg.data).get('zone')
        except ValueError:
            pass

    def on_marker(self, msg):
        if not self.markers_seen or self.markers_seen[-1][1] != msg.data:
            self.markers_seen.append((round(self.race_time(), 1), msg.data))

    # --------------------------------------------------------------- loop
    def remove_cover(self):
        req = 'name: "start_cover", position: {x: 0, y: 0, z: -10}'
        subprocess.Popen(['gz', 'service', '-s', f'/world/{self.world}/set_pose', '--reqtype',
                          'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
                          '--req', req], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def tick(self):
        if self.done or self.pose is None:
            return
        now = self.now()
        if self.t0 is None:
            self.t0 = now
        if self.t_start is None:
            if self.cover_removed is None and now - self.t0 > self.start_delay:
                if self.start_zone:
                    self.pub_start.publish(Bool(data=True))
                else:
                    self.remove_cover()
                    self.event('LIGHT', 'start light green')
                self.cover_removed = now
            elif self.cover_removed is not None and now - self.cover_removed > 10.0:
                self.event('START_DETECTOR_FAILED', 'no /race_start 10 s after green; forcing start')
                self.pub_start.publish(Bool(data=True))
            return

        x, y, z, roll, pitch, v = self.pose
        t = self.race_time()
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return self.finish('flipped', 'FLIP', f'roll {math.degrees(roll):.0f} pitch {math.degrees(pitch):.0f}')
        ground = self.course.ground(x, y, z)[0]
        if z < ground - 0.25 and z < -0.1:
            return self.finish('fell', 'FALL', f'z={z:.2f} below course surface {ground:.2f}')
        # Finish is detected per odometry message in check_finish_line().
        lt, lx, ly = self.last_progress or (t, x, y)
        if math.hypot(x - lx, y - ly) > 0.3:
            self.last_progress = (t, x, y)
        elif t - lt > 45.0:
            return self.finish('stuck', 'STUCK', 'no progress for 45 s')
        if t > self.timeout:
            return self.finish('timeout', 'TIMEOUT', f'{self.timeout:.0f} s')

    def finish(self, outcome, kind, text):
        self.event(kind, text)
        self.done = True
        order = ['start', 'ramp_up', 'bridge', 'helix', 'tunnel', 'narrow', 'gravel', 'bank_approach',
                 'bank_uturn', 'pothole', 'wide', 'buckets', 'hoops', 'car_wash']
        reached = [zn for zn in order if zn in self.zones]
        result = {'outcome': outcome, 'time_s': round(self.race_time(), 1), 'seed': self.seed,
                  'furthest_zone': reached[-1] if reached else self.zone,
                  'zones_reached': self.zones, 'max_speed': round(self.max_speed, 2),
                  'start_detector_s': (round(self.detector_fired - self.cover_removed, 2)
                                       if self.detector_fired and self.cover_removed else None),
                  'markers_seen': self.markers_seen, 'collisions': 'n/a (3D)',
                  'rails': self.course.rails, 'events': self.events}
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(self.out_dir, 'trajectory.csv'), 'w') as f:
            f.write('t,x,y,v,true_zone,mission_zone,yaw_deg,cmd_speed,cmd_steer\n')
            for row in self.traj:
                f.write(','.join(str(v) for v in row) + '\n')
        try:
            from tt_course_sim.plot import plot_run
            plot_run(self.course, self.traj, self.events, result, os.path.join(self.out_dir, 'run.png'))
        except Exception as exc:
            self.get_logger().warn(f'plot failed: {exc}')
        summary = (f'=== 3D RUN {outcome.upper()} in {result["time_s"]} s, furthest zone='
                   f'{result["furthest_zone"]}, markers={len(self.markers_seen)} ===')
        (self.get_logger().info if outcome == 'finished' else self.get_logger().error)(summary)
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = GzReferee()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
