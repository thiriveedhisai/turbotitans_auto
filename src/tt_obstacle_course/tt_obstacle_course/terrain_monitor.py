"""Terrain monitor: attitude + vibration from the ZED IMU -> terrain state and speed cap.

Publishes:
  /terrain/state      String  flat | rough | ramp_up | ramp_down | bank | tip_warn | tip_abort
  /terrain/speed_cap  Float32 m/s (large value = no cap)
  /terrain/attitude   Vector3 x=roll_deg, y=nose_up_deg, z=vibration_rms (m/s^2)
  /terrain/lidar_ok   Bool    False when tilted enough that the 2D scan hits floor/sky
"""

import collections
import math

from geometry_msgs.msg import Vector3
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, String

from tt_obstacle_course.geometry import quat_to_rpy
from tt_obstacle_course.ros_common import TTNode, run

NO_CAP = 99.0


class TerrainMonitor(TTNode):

    def __init__(self):
        super().__init__('terrain_monitor')
        self.window_s = float(self.param('vibration_window_s', 0.3))
        self.hp_hz = float(self.param('vibration_highpass_hz', 5.0))
        self.levels = [float(v) for v in self.param('vibration_rms_levels', [1.0, 2.5, 4.0])]
        self.caps = [float(v) for v in self.param('vibration_speed_caps', [1.5, 0.9, 0.5])]
        self.ramp_enter = float(self.param('ramp_pitch_enter_deg', 4.0))
        self.ramp_exit = float(self.param('ramp_pitch_exit_deg', 2.0))
        self.bank_enter = float(self.param('bank_roll_enter_deg', 5.0))
        self.bank_exit = float(self.param('bank_roll_exit_deg', 3.0))
        self.tip_warn = float(self.param('tip_roll_warn_deg', 15.0))
        self.tip_roll_abort = float(self.param('tip_roll_abort_deg', 22.0))
        self.tip_pitch_abort = float(self.param('tip_pitch_abort_deg', 25.0))
        self.tip_warn_cap = float(self.param('tip_warn_speed_cap', 0.3))
        self.lidar_gate = float(self.param('lidar_gate_tilt_deg', 4.0))
        self.hold_s = float(self.param('state_hold_s', 0.3))
        self.calib_s = float(self.param('level_calibration_s', 2.0))

        self.calib = []
        self.calib_start = None
        self.offset = None
        self.lp = None
        self.last_t = None
        self.hp_sq = collections.deque()
        self.roll = self.nose_up = self.vib = 0.0
        self.state = 'flat'
        self.candidate = 'flat'
        self.candidate_since = 0.0

        self.pub_state = self.create_publisher(String, '/terrain/state', 10)
        self.pub_cap = self.create_publisher(Float32, '/terrain/speed_cap', 10)
        self.pub_att = self.create_publisher(Vector3, '/terrain/attitude', 10)
        self.pub_lidar = self.create_publisher(Bool, '/terrain/lidar_ok', 10)
        self.create_subscription(Imu, self.param('imu_topic', '/zed/zed_node/imu/data'),
                                 self.on_imu, qos_profile_sensor_data)
        self.create_timer(0.05, self.publish)
        self.create_timer(0.5, lambda: self.pub_state.publish(String(data=self.state)))

    def on_imu(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.orientation
        roll, pitch, _ = quat_to_rpy(q.x, q.y, q.z, q.w)

        # Assume the robot starts level and still: learn the mounting offset.
        if self.offset is None:
            if self.calib_start is None:
                self.calib_start = t
            self.calib.append((roll, pitch))
            if t - self.calib_start >= self.calib_s:
                n = len(self.calib)
                self.offset = (sum(c[0] for c in self.calib) / n, sum(c[1] for c in self.calib) / n)
                self.get_logger().info(
                    f'Level calibrated: roll0={math.degrees(self.offset[0]):.1f} deg, '
                    f'pitch0={math.degrees(self.offset[1]):.1f} deg')
            return

        self.roll = math.degrees(roll - self.offset[0])
        # REP-103: positive pitch is nose down.
        self.nose_up = -math.degrees(pitch - self.offset[1])

        az = msg.linear_acceleration.z
        if self.lp is None or self.last_t is None:
            self.lp = az
        else:
            dt = max(t - self.last_t, 1e-4)
            a = 1.0 - math.exp(-2.0 * math.pi * self.hp_hz * dt)
            self.lp += a * (az - self.lp)
        self.last_t = t
        hp = az - self.lp
        self.hp_sq.append((t, hp * hp))
        while self.hp_sq and t - self.hp_sq[0][0] > self.window_s:
            self.hp_sq.popleft()
        self.vib = math.sqrt(sum(v for _, v in self.hp_sq) / len(self.hp_sq))

        self.update_state(t)

    def classify(self):
        r, p = abs(self.roll), self.nose_up
        if r > self.tip_roll_abort or abs(p) > self.tip_pitch_abort:
            return 'tip_abort'
        if r > self.tip_warn:
            return 'tip_warn'
        # Stay in a ramp/bank state until the exit threshold (hysteresis).
        ramp = self.ramp_exit if self.state in ('ramp_up', 'ramp_down') else self.ramp_enter
        if p > ramp:
            return 'ramp_up'
        if p < -ramp:
            return 'ramp_down'
        bank = self.bank_exit if self.state == 'bank' else self.bank_enter
        if r > bank:
            return 'bank'
        if self.levels and self.vib > self.levels[0]:
            return 'rough'
        return 'flat'

    def update_state(self, t):
        cand = self.classify()
        if cand != self.candidate:
            self.candidate = cand
            self.candidate_since = t
        if cand != self.state and (cand == 'tip_abort' or t - self.candidate_since >= self.hold_s):
            self.state = cand
            self.pub_state.publish(String(data=self.state))
            self.get_logger().info(
                f'terrain -> {self.state} (roll {self.roll:.1f}, nose_up {self.nose_up:.1f}, '
                f'vib {self.vib:.2f})')

    def speed_cap(self):
        cap = NO_CAP
        for level, level_cap in zip(self.levels, self.caps):
            if self.vib > level:
                cap = level_cap
        if self.state == 'tip_warn':
            cap = min(cap, self.tip_warn_cap)
        if self.state == 'tip_abort':
            cap = 0.0
        return cap

    def publish(self):
        if self.offset is None:
            return
        self.pub_cap.publish(Float32(data=float(self.speed_cap())))
        self.pub_att.publish(Vector3(x=self.roll, y=self.nose_up, z=self.vib))
        tilted = max(abs(self.roll), abs(self.nose_up)) >= self.lidar_gate
        self.pub_lidar.publish(Bool(data=not tilted))


def main(args=None):
    run(TerrainMonitor, args)
