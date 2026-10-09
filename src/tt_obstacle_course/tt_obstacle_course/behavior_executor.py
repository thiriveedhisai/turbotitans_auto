"""Behavior executor: runs the behavior requested by the mission manager.

Subscribes /behavior/cmd (JSON {"id", "behavior", "params"}), publishes
AckermannDrive on /ackermann_cmd_raw (to the speed governor) and
/behavior/status (JSON {"id", "behavior", "status", "info"}).
Behavior "idle" publishes nothing so Nav2 (via twist_to_ackermann) can drive.
"""

import json

from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from tt_obstacle_course.behaviors import (DONE, FAILED, RUNNING, Config, Context, Hold,
                                          StuckMonitor, make_behavior)
from tt_obstacle_course.geometry import OdomTracker
from tt_obstacle_course.ros_common import TTNode, dumps, odom_update, run
from tt_obstacle_course.scan_utils import ScanView

GAIN_GROUPS = ('heading_hold', 'wall_center', 'gap_follow', 'arc_turn', 'k_turn')


class BehaviorExecutor(TTNode):

    def __init__(self):
        super().__init__('behavior_executor')
        self.yaw_offset, self.flip = self.scan_args()
        self.cfg = Config(self.steering_model(), {g: self.group(g) for g in GAIN_GROUPS},
                          half_width=float(self.param('vehicle.width', 0.20)) / 2.0,
                          sweep_margin=float(self.param('sweep_margin', 0.15)))
        st = self.group('stuck')
        self.stuck = StuckMonitor(float(st.get('min_speed', 0.05)), float(st.get('time_s', 1.5)),
                                  float(st.get('backoff_distance', 0.3)), int(st.get('max_retries', 3)))
        self.rate = float(self.param('rate_hz', 20.0))

        self.odom = OdomTracker()
        self.scan = None
        self.scan_time = 0.0
        self.lidar_ok = True
        self.terrain = 'flat'
        self.pending = None
        self.behavior = None
        self.cmd_id = None
        self.name = 'idle'
        self.status = 'idle'
        self.info = ''
        self.last_t = None

        self.pub_cmd = self.create_publisher(AckermannDrive, '/ackermann_cmd_raw', 10)
        self.pub_status = self.create_publisher(String, '/behavior/status', 10)
        self.create_subscription(String, '/behavior/cmd', self.on_cmd, 10)
        self.create_subscription(Odometry, self.param('odom_topic', '/zed/zed_node/odom'),
                                 lambda m: odom_update(self.odom, m), 10)
        self.create_subscription(LaserScan, self.param('scan_topic', '/scan'),
                                 self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Bool, '/terrain/lidar_ok', self.on_lidar_ok, 10)
        self.create_subscription(String, '/terrain/state', self.on_terrain, 10)
        self.create_timer(1.0 / self.rate, self.tick)
        self.create_timer(0.2, self.publish_status)

    # -------------------------------------------------------------- inputs
    def on_cmd(self, msg):
        try:
            req = json.loads(msg.data)
            name = req['behavior']
        except (ValueError, KeyError) as exc:
            self.get_logger().error(f'bad /behavior/cmd {msg.data!r}: {exc}')
            return
        if req.get('id') == self.cmd_id and self.pending is None:
            return
        self.pending = req
        self.get_logger().info(f'behavior <- {name} {req.get("params", {})}')

    def on_scan(self, msg):
        self.scan = ScanView.from_msg(msg, self.yaw_offset, self.flip)
        self.scan_time = self.now_s()

    def on_lidar_ok(self, msg):
        self.lidar_ok = msg.data

    def on_terrain(self, msg):
        self.terrain = msg.data

    # ---------------------------------------------------------------- loop
    def context(self, now):
        fresh = self.scan is not None and now - self.scan_time < 0.5
        return Context(t=now, odom=self.odom, scan=self.scan if fresh else None,
                       lidar_ok=self.lidar_ok and fresh, terrain=self.terrain)

    def activate(self, ctx):
        req, self.pending = self.pending, None
        self.cmd_id = req.get('id')
        self.name = req['behavior']
        self.stuck.reset()
        self.info = ''
        if self.name == 'idle':
            self.behavior = None
            self.status = 'idle'
            return
        try:
            self.behavior = make_behavior(self.name, req.get('params', {}), self.cfg)
        except KeyError as exc:
            self.get_logger().error(str(exc))
            self.behavior, self.status = Hold({}, self.cfg), FAILED
            return
        self.behavior.start(ctx)
        self.status = RUNNING
        warning = getattr(self.behavior, 'warning', '')
        if warning:
            self.get_logger().warn(f'{self.name}: {warning}')

    def tick(self):
        if not self.odom.valid:
            return
        now = self.now_s()
        dt = 0.0 if self.last_t is None else now - self.last_t
        self.last_t = now
        ctx = self.context(now)
        if self.pending is not None:
            self.activate(ctx)
        if self.behavior is None:
            return

        if self.status in (DONE, FAILED):
            cmd = Hold({}, self.cfg).step(ctx, dt)
        elif self.stuck.active():
            cmd = self.stuck.step(ctx, dt)
        else:
            cmd = self.behavior.step(ctx, dt)
            if cmd.status == RUNNING and self.stuck.check(ctx, cmd, self.cfg):
                self.get_logger().warn(f'stuck in {self.name}, backing off '
                                       f'(retry {self.stuck.retries})')
                cmd = self.stuck.step(ctx, dt)
            elif self.stuck.exhausted():
                cmd.status, cmd.info = FAILED, 'stuck'
            if cmd.status != RUNNING:
                self.status = cmd.status
                self.info = cmd.info
                self.get_logger().info(f'{self.name} -> {cmd.status} {cmd.info}')
                self.publish_status()
            elif cmd.info != self.info:
                self.info = cmd.info

        self.pub_cmd.publish(AckermannDrive(speed=float(cmd.speed),
                                            steering_angle=float(cmd.steer)))

    def publish_status(self):
        self.pub_status.publish(String(data=dumps(
            {'id': self.cmd_id, 'behavior': self.name, 'status': self.status, 'info': self.info})))


def main(args=None):
    run(BehaviorExecutor, args)
