"""Odometry guard: keeps a usable pose through ZED visual-tracking dropouts.

The ZED loses visual tracking in dark or texture-poor stretches (tunnel, car-wash
ribbons); its position then stalls or stops publishing while the truck still moves.
Here a dropout is declared when the truck is commanded to move, the lidar says it is
moving, but ZED position has not changed. Until tracking returns, heading comes from
the gyro and travel from lidar scan matching (along the heading, using surfaces that
cross it); in a straight featureless lane, where the lidar can't measure travel, from
the commanded speed. Measured ~0 (e.g. pushing against a wall) stops the estimate, so
stuck recovery still works. When the ZED recovers, its pose is re-anchored to the
dead-reckoned one (no jump).

Subscribes: zed_odom_topic, imu_topic, /ackermann_cmd, scan_topic
Publishes:  /odom/fused (Odometry, used by the stack), /odom/degraded (Bool),
            TF odom -> zed_camera_link (the ZED wrapper's own TF is disabled)
"""

import collections
import json
import math

import numpy as np
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from tt_obstacle_course.geometry import quat_to_rpy, wrap_angle
from tt_obstacle_course.ros_common import TTNode, run
from tt_obstacle_course.scan_utils import ScanView, scan_shift


class OdomGuard(TTNode):

    def __init__(self):
        super().__init__('odom_guard')
        g = self.param
        self.freeze_s = float(g('freeze_s', 0.5))
        self.stale_s = float(g('stale_s', 0.3))
        self.cmd_min = float(g('cmd_min_speed', 0.12))
        self.speed_scale = float(g('dead_reckon_speed_scale', 0.85))
        self.freeze_m = float(g('freeze_distance', 0.01))
        self.resume_m = float(g('resume_distance', 0.03))
        self.scan_dt = float(g('scan_compare_s', 0.5))
        self.scan_delta = float(g('scan_change_m', 0.04))
        self.scan_frac = float(g('scan_moving_fraction', 0.03))
        # Heading aid: the course is rectilinear, so straight parallel lane walls give the
        # true heading modulo 90 deg; pull the yaw onto that grid (position untouched).
        self.wall_tol = math.radians(float(g('wall_heading_tol_deg', 8.0)))
        self.wall_gain = float(g('wall_heading_gain', 0.08))
        self.grid = None                # learned grid phase (start lane), rad
        self.yaw_off, self.flip = self.scan_args()

        self.raw = None                 # (t, x, y, z, roll, pitch, yaw)
        self.raw_hist = collections.deque()
        self.raw_t = None
        self.offset = (0.0, 0.0, 0.0)   # fused = offset o raw (SE2)
        self.fused = None               # [x, y, z, roll, pitch, yaw]
        self.degraded = False
        self.cmd = AckermannDrive()
        self.cmd_since = None
        self.gyro_z = 0.0
        # Heading integrated on every IMU message by its own stamp. Sampling the latest
        # rate once per tick lost rotation whenever ticks stretched under CPU load
        # (26 deg over the hoop turns in a 3-D run -> car wash driven into a wall).
        self.gyro_yaw = 0.0
        self.gyro_used = 0.0
        self.imu_t = None
        self.imu_rp = (0.0, 0.0)
        self.scans = collections.deque()
        self.scan_moving = False
        self.lidar_speed = None
        self.distance_dr = 0.0
        self.raw_speed = 0.0
        self.dr_speed = 0.0
        self.last_t = None

        self.pub = self.create_publisher(Odometry, '/odom/fused', 10)
        self.pub_flag = self.create_publisher(Bool, '/odom/degraded', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, self.param('zed_odom_topic', '/zed/zed_node/odom'),
                                 self.on_raw, 20)
        self.create_subscription(Imu, self.param('imu_topic', '/zed/zed_node/imu/data'),
                                 self.on_imu, qos_profile_sensor_data)
        self.create_subscription(AckermannDrive, '/ackermann_cmd', self.on_cmd, 10)
        self.create_subscription(LaserScan, self.param('scan_topic', '/scan'), self.on_scan,
                                 qos_profile_sensor_data)
        self.tag_gain = float(g('tag_heading_gain', 0.04))
        self.tag_range = float(g('tag_heading_max_range', 2.5))
        self.create_subscription(String, '/marker/detections', self.on_tags, 10)
        self.create_timer(1.0 / 30.0, self.tick)
        self.create_timer(0.1, lambda: self.pub_flag.publish(Bool(data=self.degraded)))

    # ------------------------------------------------------------ inputs
    def on_raw(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        t = self.now_s()
        self.raw_speed = msg.twist.twist.linear.x
        self.raw = (t, p.x, p.y, p.z, roll, pitch, yaw)
        self.raw_t = t
        self.raw_hist.append((t, p.x, p.y))
        while self.raw_hist and t - self.raw_hist[0][0] > 2.0:
            self.raw_hist.popleft()
        if self.fused is None:
            self.fused = [p.x, p.y, p.z, roll, pitch, yaw]

    def on_imu(self, msg):
        q = msg.orientation
        roll, pitch, _ = quat_to_rpy(q.x, q.y, q.z, q.w)
        self.imu_rp = (roll, pitch)
        # Body yaw rate projected on the vertical (fine for the course's modest tilts).
        self.gyro_z = (msg.angular_velocity.z * math.cos(roll) * math.cos(pitch)
                       + msg.angular_velocity.y * math.sin(roll) - msg.angular_velocity.x * math.sin(pitch))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.imu_t is not None and 0.0 < t - self.imu_t < 0.5:
            self.gyro_yaw += self.gyro_z * (t - self.imu_t)
        self.imu_t = t

    def on_cmd(self, msg):
        moving = abs(msg.speed) >= self.cmd_min
        if moving and self.cmd_since is None:
            self.cmd_since = self.now_s()
        elif not moving:
            self.cmd_since = None
        self.cmd = msg

    def on_scan(self, msg):
        t = self.now_s()
        view = ScanView.from_msg(msg, self.yaw_off, self.flip)
        r = np.asarray(view.ranges, dtype=np.float32)
        yaw = self.fused[5] if self.fused is not None else 0.0
        self.scans.append((t, r, view.angles, yaw))
        while len(self.scans) > 2 and t - self.scans[1][0] >= self.scan_dt:
            self.scans.popleft()
        t0, r0, a0, yaw0 = self.scans[0]
        if t - t0 < self.scan_dt * 0.8 or r0.shape != r.shape:
            return
        ok = np.isfinite(r) & np.isfinite(r0) & (r > 0.2) & (r < 8.0) & (r0 > 0.2) & (r0 < 8.0)
        if ok.sum() < 20:
            return
        # Measured travel when the scene allows it (anything across the heading: wall
        # ends, corners, posts, a wall we're pushing against). Rocking against a wall
        # changes many beams but measures ~0, so it no longer reads as motion.
        shift = scan_shift((a0, r0), (view.angles, r), dyaw=wrap_angle(yaw - yaw0))
        self.lidar_speed = None if shift is None else shift / (t - t0)
        if self.lidar_speed is not None:
            self.scan_moving = abs(self.lidar_speed) > 0.05
        else:
            # Straight featureless lane: can't measure, only tell the scene changed.
            # Compare after undoing the rotation: pivoting in place while wedged changes
            # most beams and read as motion (3 m of phantom travel skipped the tunnel).
            inc = float(msg.angle_increment) or 1.0
            k = int(round(wrap_angle(yaw - yaw0) / inc))
            r0 = np.roll(r0, -k)
            ok = np.isfinite(r) & np.isfinite(r0) & (r > 0.2) & (r < 8.0) & (r0 > 0.2) & (r0 < 8.0)
            self.scan_moving = (ok.sum() >= 20 and
                                float(np.mean(np.abs(r[ok] - r0[ok]) > self.scan_delta)) > self.scan_frac)
        self.wall_heading(msg)

    def wall_heading(self, msg):
        # Only on a straight: a short window of a curved lane (helix) also looks straight.
        if (self.fused is None or abs(self.gyro_z) > 0.05
                or max(abs(self.imu_rp[0]), abs(self.imu_rp[1])) > math.radians(3.0)):
            return
        walls = ScanView.from_msg(msg, self.yaw_off, self.flip).lane_walls(
            max_half_width=1.2, x_min=-1.0, x_max=2.0)
        left, right = walls.get(1), walls.get(-1)
        if not (left and right) or min(left[2], right[2]) < 15:
            return
        if abs(left[1] - right[1]) > 0.05 or max(abs(left[3]), abs(right[3])) > 0.05:
            return
        if not 0.4 < left[0] - right[0] < 2.4:
            return
        lane = wrap_angle(self.fused[5] + math.atan((left[1] + right[1]) / 2))
        quarter = math.pi / 2
        if self.grid is None:
            phase = math.remainder(lane, quarter)
            if abs(phase) < self.wall_tol:
                self.grid = phase
                self.get_logger().info(f'course grid phase {math.degrees(phase):+.1f} deg from the start lane')
            return
        err = math.remainder(self.grid - lane, quarter)
        if abs(err) > self.wall_tol:
            return
        self.fused[5] = wrap_angle(self.fused[5] + self.wall_gain * err)
        if not self.degraded and self.raw is not None:
            self.reanchor()

    def on_tags(self, msg):
        """A tag on a course wall gives the wall's direction, i.e. heading mod 90 deg.
        Only from good views: an oblique tag's PnP normal is noisy (flip ambiguity) and
        once pulled the heading 13 deg off at the car-wash entry."""
        if self.fused is None or self.grid is None or abs(self.gyro_z) > 0.05:
            return
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        for d in dets:
            if 'normal_deg' not in d or d.get('distance', 99.0) > self.tag_range:
                continue
            # Facing us within 35 deg (normal points back toward the camera, +-180).
            if abs(abs(d['normal_deg']) - 180.0) > 35.0 and abs(d['normal_deg']) > 35.0:
                continue
            face = wrap_angle(self.fused[5] + math.radians(d['normal_deg']))
            err = math.remainder(self.grid - face, math.pi / 2)
            if abs(err) > self.wall_tol:
                continue
            self.fused[5] = wrap_angle(self.fused[5] + self.tag_gain * err)
            if not self.degraded and self.raw is not None:
                self.reanchor()

    # ------------------------------------------------------------ logic
    def raw_moved(self, window):
        if not self.raw_hist:
            return 0.0
        t, x, y = self.raw_hist[-1]
        old = next(((tt, xx, yy) for tt, xx, yy in self.raw_hist if t - tt <= window), self.raw_hist[0])
        return math.hypot(x - old[1], y - old[2])

    def apply_offset(self):
        ox, oy, oyaw = self.offset
        _, x, y, z, roll, pitch, yaw = self.raw
        c, s = math.cos(oyaw), math.sin(oyaw)
        self.fused = [ox + c * x - s * y, oy + s * x + c * y, z, roll, pitch, wrap_angle(yaw + oyaw)]

    def reanchor(self):
        """Choose the offset so that offset o raw == current fused pose."""
        _, x, y, _, _, _, yaw = self.raw
        fx, fy, _, _, _, fyaw = self.fused
        oyaw = wrap_angle(fyaw - yaw)
        c, s = math.cos(oyaw), math.sin(oyaw)
        self.offset = (fx - (c * x - s * y), fy - (s * x + c * y), oyaw)

    def tick(self):
        if self.fused is None:
            return
        t = self.now_s()
        dt = 0.0 if self.last_t is None else min(0.5, t - self.last_t)
        self.last_t = t
        commanded = self.cmd_since is not None and t - self.cmd_since >= self.freeze_s
        stale = self.raw_t is None or t - self.raw_t > self.stale_s
        frozen = self.raw_moved(self.freeze_s) < self.freeze_m
        dyaw = self.gyro_yaw - self.gyro_used
        self.gyro_used = self.gyro_yaw

        if not self.degraded:
            if commanded and self.scan_moving and (stale or frozen):
                self.degraded = True
                # The stall began freeze_s ago: credit the distance driven since.
                self.dead_reckon(self.freeze_s)
                self.get_logger().warn('ZED tracking lost: dead-reckoning on gyro + commanded speed')
            else:
                self.apply_offset()
        else:
            if not stale and self.raw_moved(0.3) > self.resume_m:
                # VIO heading runs on through a visual dropout (IMU): resume on it unless it
                # jumped (relocalisation), so dead-reckoned yaw error isn't locked in.
                vio_yaw = wrap_angle(self.raw[6] + self.offset[2])
                if abs(wrap_angle(vio_yaw - self.fused[5])) < math.radians(20.0):
                    self.fused[5] = vio_yaw
                self.reanchor()
                self.degraded = False
                self.get_logger().info(f'ZED tracking back after {self.distance_dr:.2f} m dead-reckoned')
                self.distance_dr = 0.0
                self.apply_offset()
            else:
                self.dead_reckon(dt, dyaw)
        self.publish()

    def dead_reckon(self, dt, dyaw=None):
        if dyaw is not None:
            self.fused[5] = wrap_angle(self.fused[5] + dyaw)
        self.fused[3], self.fused[4] = self.imu_rp
        if self.lidar_speed is not None:
            v = self.lidar_speed
        else:
            v = self.cmd.speed * self.speed_scale if self.scan_moving else 0.0
        self.dr_speed = v
        self.fused[0] += v * dt * math.cos(self.fused[5])
        self.fused[1] += v * dt * math.sin(self.fused[5])
        self.distance_dr += abs(v) * dt

    def publish(self):
        x, y, z, roll, pitch, yaw = self.fused
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'odom'
        out.child_frame_id = 'zed_camera_link'
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = x, y, z
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        o = out.pose.pose.orientation
        o.w = cr * cp * cy + sr * sp * sy
        o.x = sr * cp * cy - cr * sp * sy
        o.y = cr * sp * cy + sr * cp * sy
        o.z = cr * cp * sy - sr * sp * cy
        out.twist.twist.linear.x = float(self.dr_speed if self.degraded else self.raw_speed)
        self.pub.publish(out)
        tf = TransformStamped()
        tf.header = out.header
        tf.child_frame_id = out.child_frame_id
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, z
        tf.transform.rotation = o
        self.tf.sendTransform(tf)


def main(args=None):
    run(OdomGuard, args)
