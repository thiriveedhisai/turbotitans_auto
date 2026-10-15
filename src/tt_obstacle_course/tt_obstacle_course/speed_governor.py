"""Speed governor: last stage before the motor controller.

/ackermann_cmd_raw -> limits -> /ackermann_cmd at a fixed rate:
  * zone cap (/mission/speed_cap) x global_speed_scale, terrain cap, max speeds
  * lateral-acceleration (rollover) cap from the commanded curvature
  * front/rear collision slow-down and stop from the lidar (distances measured
    from the lidar centre), with optional front blind zone for the car wash
  * acceleration and steering-rate limits, e-stop
  * /teleop_cmd (tt_dash drive panel, keyboard teleop) overrides the stack while it keeps
    arriving; every limit above except the zone caps still applies
"""

import json
import math

from ackermann_msgs.msg import AckermannDrive
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

from tt_obstacle_course.geometry import clamp
from tt_obstacle_course.ros_common import TTNode, run
from tt_obstacle_course.scan_utils import DEG


class SpeedGovernor(TTNode):

    def __init__(self):
        super().__init__('speed_governor')
        p = self.param
        self.scale = float(p('global_speed_scale', 1.0))
        self.max_speed = float(p('max_speed', 2.5))
        self.max_rev = float(p('max_reverse_speed', 0.6))
        self.max_accel = float(p('max_accel', 1.5))
        self.max_decel = float(p('max_decel', 3.0))
        self.max_lat = float(p('max_lateral_accel', 2.5))
        self.steer_rate = float(p('steer_rate_limit', 3.0))
        self.front_half = float(p('front_half_angle_deg', 35.0)) * DEG
        self.corridor = float(p('vehicle.width', 0.20)) / 2.0 + float(p('corridor_margin', 0.08))
        self.body_half = float(p('vehicle.width', 0.20)) / 2.0
        self.front_overhang = float(p('vehicle.lidar_to_front', 0.20))
        self.rear_overhang = float(p('vehicle.lidar_to_rear', 0.30))
        # Rotation centre for the swept-arc check: rear axle, this far behind the lidar.
        self.axle_back = float(p('vehicle.lidar_ahead_of_axle', 0.23))
        self.slow_d = float(p('slow_distance', 0.8))
        self.stop_d = float(p('stop_distance', 0.33))
        self.rear_stop_d = float(p('rear_stop_distance', 0.33))
        self.ttc = float(p('ttc_brake_s', 0.6))
        self.crawl = float(p('crawl_speed', 0.15))
        self.degraded_cap = float(p('degraded_speed_cap', 0.4))
        self.degraded = False
        self.timeout = float(p('cmd_timeout', 0.3))
        rate = float(p('rate_hz', 20.0))
        self.steering = self.steering_model()

        self.raw = AckermannDrive()
        self.raw_t = -1e9
        self.terrain_cap = 99.0
        self.mission_cap = 99.0
        self.ignore_front = 0.0
        self.lidar_ok = True
        self.estop = False
        self.scan = None
        self.scan_t = -1e9
        self.out_v = 0.0
        self.out_s = 0.0
        self.dt = 1.0 / rate

        self.pub = self.create_publisher(AckermannDrive, '/ackermann_cmd', 10)
        self.create_subscription(AckermannDrive, '/ackermann_cmd_raw', self.on_raw, 10)
        self.teleop = AckermannDrive()
        self.teleop_t = -1e9
        self.teleop_max = float(p('teleop_max_speed', 1.0))
        self.teleop_collision = bool(p('teleop_collision_stop', True))
        self.manual = False
        self.create_subscription(AckermannDrive, '/teleop_cmd', self.on_teleop, 10)
        self.create_subscription(Float32, '/terrain/speed_cap', self.on_terrain_cap, 10)
        self.create_subscription(Float32, '/mission/speed_cap', self.on_mission_cap, 10)
        self.create_subscription(Bool, '/terrain/lidar_ok', self.on_lidar_ok, 10)
        self.create_subscription(Bool, '/estop', self.on_estop, 10)
        # Hardware e-stop (estop_bridge). Required: no message within hw_estop_stale_s = stop.
        # Latched once it goes RUN -> STOP; cleared by /estop false (dashboard Release) when RUN again.
        self.hw_required = bool(p('hw_estop_required', False))
        self.hw_latch = bool(p('hw_estop_latch', True))
        self.hw_stale = float(p('hw_estop_stale_s', 1.0))
        self.hw_stop = True
        self.hw_t = -1e9
        self.hw_seen_run = False
        self.hw_latched = False
        self.create_subscription(Bool, '/estop/hw', self.on_hw_estop, 10)
        self.create_subscription(Bool, '/odom/degraded', self.on_degraded, 10)
        self.terrain = 'flat'
        self.create_subscription(String, '/terrain/state',
                                 lambda m: setattr(self, 'terrain', m.data), 10)
        self.ribbon_zones = tuple(p('ribbon_zones', ['car_wash', 'finish']))
        self.zone = None
        self.create_subscription(String, '/mission/state', self.on_mission_state, 10)
        self.create_subscription(String, '/behavior/cmd', self.on_behavior, 10)
        self.create_subscription(LaserScan, p('scan_topic', '/scan'), self.on_scan,
                                 qos_profile_sensor_data)
        self.create_timer(self.dt, self.tick)

    def on_raw(self, msg):
        self.raw, self.raw_t = msg, self.now_s()

    def on_teleop(self, msg):
        if self.now_s() - self.teleop_t > 1.0:
            self.get_logger().warn('teleop: manual control')
        self.teleop, self.teleop_t = msg, self.now_s()

    def on_terrain_cap(self, msg):
        self.terrain_cap = msg.data

    def on_mission_cap(self, msg):
        self.mission_cap = msg.data

    def on_mission_state(self, msg):
        try:
            self.zone = json.loads(msg.data).get('zone')
        except ValueError:
            pass

    def on_degraded(self, msg):
        self.degraded = msg.data

    def on_lidar_ok(self, msg):
        self.lidar_ok = msg.data

    def on_estop(self, msg):
        if msg.data and not self.estop:
            self.get_logger().warn('E-STOP')
        self.estop = msg.data
        if not msg.data and self.hw_latched:
            if self.hw_stop:
                self.get_logger().warn('E-STOP: hardware button still at STOP, not released')
            else:
                self.hw_latched = False
                self.get_logger().warn('E-STOP: hardware latch released')

    def on_hw_estop(self, msg):
        self.hw_t = self.now_s()
        if msg.data and not self.hw_stop and self.hw_seen_run and self.hw_latch:
            self.hw_latched = True
            self.get_logger().warn('E-STOP: hardware STOP latched (Release on the dashboard to clear)')
        self.hw_seen_run = self.hw_seen_run or not msg.data
        self.hw_stop = msg.data

    def stopped(self, now):
        if self.estop or self.hw_latched:
            return True
        if now - self.hw_t > self.hw_stale:
            return self.hw_required
        return self.hw_stop

    def on_behavior(self, msg):
        try:
            params = json.loads(msg.data).get('params', {})
        except ValueError:
            return
        self.ignore_front = float(params.get('ignore_front_deg', 0.0)) * DEG

    def on_scan(self, msg):
        self.scan = self.scan_view(msg)
        self.scan_t = self.now_s()

    def limit(self, v, s, now, manual=False):
        cap = min(self.max_speed * self.scale, self.mission_cap * self.scale, self.terrain_cap)
        if manual:
            # No mission cap: before the race start it is 0.
            cap = min(self.max_speed, self.teleop_max, self.terrain_cap)
        # Not while climbing: tracking drops when nose-up at the crest, and the cap then
        # held a low-friction stall below the speed needed to get over.
        if self.degraded and self.terrain not in ('ramp_up', 'bank'):
            cap = min(cap, self.degraded_cap)
        v = min(v, cap) if v >= 0 else max(v, -min(self.max_rev, cap))

        kappa = abs(self.steering.steer_to_curvature(s))
        if kappa > 1e-3:
            v_lat = math.sqrt(self.max_lat / kappa)
            v = clamp(v, -v_lat, v_lat)

        if self.lidar_ok and self.scan is not None and now - self.scan_t < 0.5 \
                and (self.teleop_collision or not manual):
            # Check the band swept at the steering the wheels actually have now (the
            # servo moves even while stopped), not a straight strip: a wall alongside
            # while steering away from it is not an obstacle.
            k = self.steering.steer_to_curvature(self.out_s)
            if v > 0:
                # front_half_angle_deg only defines how wide the car-wash blind cone may get.
                if self.ignore_front < self.front_half:
                    d = self.scan.corridor_clearance(self.corridor, True, self.ignore_front,
                                                     min_x=self.front_overhang, curvature=k,
                                                     axle_back=self.axle_back, body_half=self.body_half)
                    v = self.collision_cap(v, d, self.stop_d)
            elif v < 0 and self.zone not in self.ribbon_zones:
                # Not under ribbons: they hang behind as well and zeroed every back-off
                # (wedged 0.44 m short of the finish line).
                d = self.scan.corridor_clearance(self.corridor, False, min_x=self.rear_overhang,
                                                 curvature=k, axle_back=self.axle_back,
                                                 body_half=self.body_half)
                v = -self.collision_cap(-v, d, self.rear_stop_d)
        return v

    def collision_cap(self, v, d, stop_d):
        if d <= stop_d:
            return 0.0
        if d < self.slow_d:
            frac = (d - stop_d) / (self.slow_d - stop_d)
            v = min(v, max(self.crawl, v * frac))
        return min(v, max(self.crawl, (d - stop_d) / self.ttc))

    def tick(self):
        now = self.now_s()
        manual = now - self.teleop_t <= self.timeout
        if self.manual and not manual:
            self.get_logger().warn('teleop: released, stack in control')
        self.manual = manual
        cmd, cmd_t = (self.teleop, self.teleop_t) if manual else (self.raw, self.raw_t)
        estop = self.stopped(now)
        if estop or now - cmd_t > self.timeout:
            v_target, s_target = 0.0, self.out_s
        else:
            v_target = self.limit(cmd.speed, cmd.steering_angle, now, manual)
            s_target = cmd.steering_angle
        speeding_up = abs(v_target) > abs(self.out_v) and v_target * self.out_v >= 0
        lim = (self.max_accel if speeding_up else self.max_decel) * self.dt
        self.out_v = 0.0 if estop else self.out_v + clamp(v_target - self.out_v, -lim, lim)
        self.out_s += clamp(s_target - self.out_s, -self.steer_rate * self.dt, self.steer_rate * self.dt)
        self.pub.publish(AckermannDrive(speed=float(self.out_v), steering_angle=float(self.out_s)))


def main(args=None):
    run(SpeedGovernor, args)
