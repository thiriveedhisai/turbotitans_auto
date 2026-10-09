"""Course perception from lidar: wall openings (gaps) and hoop gates.

Publishes (odom frame, only after a detection persists for N scans):
  /perception/left_gap    PoseStamped  opening in the left wall, yaw = through it
  /perception/right_gap   PoseStamped
  /perception/gate        PoseStamped  nearest hoop gate, yaw = passing direction
"""

import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from tt_obstacle_course.geometry import OdomTracker, wrap_angle, yaw_to_quat
from tt_obstacle_course.ros_common import TTNode, odom_update, run
from tt_obstacle_course.scan_utils import ScanView


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
        self.yaw_offset, self.flip = self.scan_args()
        self.laser_x = float(self.param('laser_offset_x', 0.0))
        self.laser_y = float(self.param('laser_offset_y', 0.0))
        width = float(self.param('vehicle.width', 0.20))
        self.min_gap = width + float(self.param('min_gap_width_margin', 0.25))
        self.gap_depth = float(self.param('min_gap_depth_beyond', 1.5))
        # Openings further back than this (e.g. the corridor just left) are ignored.
        self.gap_max_angle = math.radians(float(self.param('gap_max_angle_deg', 110.0)))
        self.gate_width = float(self.param('gate_width', 0.60))
        self.gate_tol = float(self.param('hoop_post_spacing_tol', 0.10))
        self.post_extent = float(self.param('post_max_extent', 0.15))
        self.frame = self.param('odom_frame', 'odom')
        persist = int(self.param('persist_scans', 3))
        match = float(self.param('match_distance', 0.3))

        self.odom = OdomTracker()
        self.lidar_ok = True
        self.tracks = {'left_gap': Track(persist, match), 'right_gap': Track(persist, match),
                       'gate': Track(persist, match)}
        self.pubs = {k: self.create_publisher(PoseStamped, f'/perception/{k}', 10) for k in self.tracks}

        self.create_subscription(Odometry, self.param('odom_topic', '/zed/zed_node/odom'),
                                 lambda m: odom_update(self.odom, m), 10)
        self.create_subscription(Bool, '/terrain/lidar_ok', self.on_lidar_ok, 10)
        self.create_subscription(LaserScan, self.param('scan_topic', '/scan'),
                                 self.on_scan, qos_profile_sensor_data)

    def on_lidar_ok(self, msg):
        self.lidar_ok = msg.data

    def to_odom(self, det):
        ox, oy = self.odom.to_odom(det['x'] + self.laser_x, det['y'] + self.laser_y)
        return ox, oy, wrap_angle(self.odom.yaw + det['yaw'])

    def on_scan(self, msg):
        if not self.odom.valid or not self.lidar_ok:
            return
        scan = ScanView.from_msg(msg, self.yaw_offset, self.flip)
        found = {
            'left_gap': scan.find_side_openings(+1, self.min_gap, self.gap_depth,
                                                a_to=self.gap_max_angle),
            'right_gap': scan.find_side_openings(-1, self.min_gap, self.gap_depth,
                                                 a_to=self.gap_max_angle),
            'gate': scan.find_gates(self.gate_width, self.gate_tol, self.post_extent),
        }
        for key, dets in found.items():
            pose = self.tracks[key].update(self.to_odom(dets[0]) if dets else None)
            if pose is not None:
                self.publish(key, pose, msg.header.stamp)

    def publish(self, key, pose, stamp):
        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.frame
        out.pose.position.x, out.pose.position.y = pose[0], pose[1]
        qx, qy, qz, qw = yaw_to_quat(pose[2])
        out.pose.orientation.x, out.pose.orientation.y = qx, qy
        out.pose.orientation.z, out.pose.orientation.w = qz, qw
        self.pubs[key].publish(out)


def main(args=None):
    run(CoursePerception, args)
