"""Course perception from lidar + ZED depth: wall openings (gaps) and hoop gates.

Publishes (odom frame, only after a detection persists for N scans):
  /perception/left_gap    PoseStamped  opening in the left wall, yaw = through it
  /perception/right_gap   PoseStamped
  /perception/front_gap   PoseStamped  opening in the wall ahead (bucket-field exit)
  /perception/gate        PoseStamped  nearest hoop gate, yaw = passing direction;
                                       position.z = 1 if depth confirmed it, else 0

A hoop gate is a lidar pair of thin legs; the depth image can confirm the hoop
frame (a top spanning the opening at the same distance, see-through below it), in
which case the mission takes it as sensed instead of checking where it should be.
"""

import math

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Bool

from tt_obstacle_course.depth_check import DepthView, leg_points
from tt_obstacle_course.geometry import OdomTracker, wrap_angle, yaw_to_quat
from tt_obstacle_course.ros_common import TTNode, odom_update, run


class Track:
    """Confirms a detection that stays in the same odom spot across scans."""

    def __init__(self, persist, match_dist):
        self.persist = persist
        self.match_dist = match_dist
        self.pose = None
        self.count = 0

    def update(self, pose):
        if pose is None:
            self.count = max(0, self.count - 1)
            return None
        if self.pose and math.hypot(pose[0] - self.pose[0], pose[1] - self.pose[1]) < self.match_dist:
            # Smooth toward the new estimate.
            self.pose = tuple(0.7 * a + 0.3 * b for a, b in zip(self.pose[:2], pose[:2])) + (pose[2],)
            self.count += 1
        else:
            self.pose = pose
            self.count = 1
        return self.pose if self.count >= self.persist else None


class CoursePerception(TTNode):

    def __init__(self):
        super().__init__('course_perception')
        self.laser_x = float(self.param('laser_offset_x', 0.0))
        self.laser_y = float(self.param('laser_offset_y', 0.0))
        width = float(self.param('vehicle.width', 0.20))
        self.min_gap = width + float(self.param('min_gap_width_margin', 0.25))
        self.gap_depth = float(self.param('min_gap_depth_beyond', 1.5))
        # Openings further back than this (e.g. the corridor just left) are ignored.
        self.gap_max_angle = math.radians(float(self.param('gap_max_angle_deg', 110.0)))
        # Wider "openings" are far wall stretches dropping out of range seen obliquely
        # (read 2.5-3 m wide, up to 0.9 m off the real 36" entrance); so are far ones
        # (>3.9 m: sparse beams through the doorway onto the field). The real one
        # confirms at 2-3 m from the search line.
        self.side_gap_max = float(self.param('side_gap_max_width', 1.4))
        self.side_gap_range = float(self.param('side_gap_max_range', 3.5))
        self.gate_width = float(self.param('gate_width', 0.60))
        self.front_gap = (float(self.param('front_gap_min_width', 0.6)),
                          float(self.param('front_gap_max_width', 1.4)),
                          float(self.param('front_gap_depth_beyond', 1.0)))
        self.gate_tol = float(self.param('hoop_post_spacing_tol', 0.10))
        self.post_extent = float(self.param('post_max_extent', 0.15))
        self.post_isolation = float(self.param('post_isolation', 0.12))
        self.frame = self.param('odom_frame', 'odom')
        persist = int(self.param('persist_scans', 3))
        match = float(self.param('match_distance', 0.3))

        self.odom = OdomTracker()
        self.lidar_ok = True
        self.tracks = {'left_gap': Track(persist, match), 'right_gap': Track(persist, match),
                       'front_gap': Track(persist, match), 'gate': Track(persist, match)}
        self.pubs = {k: self.create_publisher(PoseStamped, f'/perception/{k}', 10) for k in self.tracks}

        self.create_subscription(Odometry, self.param('odom_topic', '/zed/zed_node/odom'),
                                 lambda m: odom_update(self.odom, m), 10)
        self.create_subscription(Bool, '/terrain/lidar_ok', self.on_lidar_ok, 10)
        self.create_subscription(LaserScan, self.param('scan_topic', '/scan'),
                                 self.on_scan, qos_profile_sensor_data)
        self.cam_h = float(self.param('camera_height', 0.20))
        self.K = None
        self.depth = None                # (stamp_s, image)
        self.gate_confirmed = False
        self.create_subscription(CameraInfo, self.param('camera_info_topic',
                                                        '/zed/zed_node/rgb/color/rect/camera_info'),
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.param('depth_topic', '/zed/zed_node/depth/depth_registered'),
                                 self.on_depth, qos_profile_sensor_data)

    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_depth(self, msg):
        if msg.encoding != '32FC1':
            return
        img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self.depth = (self.now_s(), img)

    def check_gates(self, gates):
        """Depth verdict per lidar gate. Depth only adds confidence: a confirmed hoop is
        taken as sensed; anything else stays a lidar candidate for the mission's
        expected-position check (a depth veto lost real hoops seen obliquely)."""
        if self.K is None or self.depth is None or self.now_s() - self.depth[0] > 0.3:
            return [(g, False) for g in gates]
        view = DepthView(self.depth[1], self.K, self.cam_h)
        out = []
        for g in gates:
            legs = [(x + self.laser_x, y + self.laser_y) for x, y in leg_points(g)]
            out.append((g, view.hoop_frame(legs) is True))
        # Prefer a confirmed hoop over a nearer unconfirmed pair.
        return sorted(out, key=lambda gc: not gc[1])

    def on_lidar_ok(self, msg):
        self.lidar_ok = msg.data

    def to_odom(self, det):
        ox, oy = self.odom.to_odom(det['x'] + self.laser_x, det['y'] + self.laser_y)
        return ox, oy, wrap_angle(self.odom.yaw + det['yaw'])

    def on_scan(self, msg):
        if not self.odom.valid or not self.lidar_ok:
            return
        scan = self.scan_view(msg)
        lo, hi, depth = self.front_gap
        found = {
            side: [g for g in scan.find_side_openings(s, self.min_gap, self.gap_depth,
                                                     a_to=self.gap_max_angle)
                   if g['width'] <= self.side_gap_max
                   and math.hypot(g['x'], g['y']) <= self.side_gap_range]
            for side, s in (('left_gap', +1), ('right_gap', -1))
        }
        found.update({
            'front_gap': [g for g in scan.find_front_openings(lo, depth) if g['width'] <= hi],
            'gate': scan.find_gates(self.gate_width, self.gate_tol, self.post_extent,
                                    isolation=self.post_isolation),
        })
        for key, dets in found.items():
            if key == 'gate':
                checked = self.check_gates(dets)
                self.gate_confirmed = bool(checked) and checked[0][1]
                dets = [g for g, _ in checked]
            pose = self.tracks[key].update(self.to_odom(dets[0]) if dets else None)
            if pose is not None:
                self.publish(key, pose, msg.header.stamp)

    def publish(self, key, pose, stamp):
        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.frame
        out.pose.position.x, out.pose.position.y = pose[0], pose[1]
        out.pose.position.z = 1.0 if key == 'gate' and self.gate_confirmed else 0.0
        qx, qy, qz, qw = yaw_to_quat(pose[2])
        out.pose.orientation.x, out.pose.orientation.y = qx, qy
        out.pose.orientation.z, out.pose.orientation.w = qz, qw
        self.pubs[key].publish(out)


def main(args=None):
    run(CoursePerception, args)
