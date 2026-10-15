"""Fiducial marker detector (AprilTag 36h11 by default, via OpenCV aruco).

Publishes:
  /zone_event         String  zone name for a confirmed marker (rate limited)
  /marker/detections  String  JSON list of {id, zone, distance, bearing_deg}
  /perception/bright_exit  Float32  bearing (rad, + left) of a bright lane end (tunnel exit),
                      NaN when nothing stands out
  /marker/debug       String  JSON {w, h, tags: [... corners, status], rejected: [quads]}, only
                      while someone subscribes (tt_dash camera view)
"""

import math

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String

from tt_obstacle_course.light_blob import bright_exit
from tt_obstacle_course.ros_common import TTNode, dumps, run

FAMILIES = {
    'tag36h11': 'DICT_APRILTAG_36h11',
    'tag25h9': 'DICT_APRILTAG_25h9',
    'tag16h5': 'DICT_APRILTAG_16h5',
    'aruco_4x4_50': 'DICT_4X4_50',
    'aruco_5x5_50': 'DICT_5X5_50',
    'aruco_6x6_50': 'DICT_6X6_50',
}


class MarkerDetector(TTNode):

    def __init__(self):
        super().__init__('marker_detector')
        family = self.param('family', 'tag36h11')
        if not hasattr(cv2, 'aruco'):
            raise RuntimeError('cv2.aruco not available; install OpenCV >= 4.7 or opencv-contrib')
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, FAMILIES[family]))
        if hasattr(cv2.aruco, 'ArucoDetector'):
            det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
            self.detect = det.detectMarkers
        else:
            params = cv2.aruco.DetectorParameters_create()
            self.detect = lambda g: cv2.aruco.detectMarkers(g, dictionary, parameters=params)

        self.size = float(self.param('tag_size', 0.20))
        self.max_range = float(self.param('max_range', 6.0))
        # Zone events only fire this close, so a far marker can't switch zones early.
        self.trigger_range = float(self.param('trigger_range', 1.5))
        # Launch tags:=false turns zone events off (detections still published for the dashboard).
        if not bool(self.param('zone_events', True)):
            self.trigger_range = -1.0
        self.min_side = float(self.param('min_side_px', 12.0))
        self.confirm = int(self.param('confirm_frames', 2))
        self.repeat_s = float(self.param('repeat_period_s', 2.0))
        ids = [int(i) for i in self.param('tag_ids', [0])]
        zones = list(self.param('tag_zones', ['start']))
        self.zone_of = dict(zip(ids, zones))

        s = self.size / 2.0
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        self.K = None
        self.D = None
        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.hits = {}
        self.last_sent = {}

        self.pub_zone = self.create_publisher(String, '/zone_event', 10)
        self.pub_det = self.create_publisher(String, '/marker/detections', 10)
        self.pub_debug = self.create_publisher(String, '/marker/debug', 2)
        self.pub_bright = self.create_publisher(Float32, '/perception/bright_exit', 10)
        self.bright_ratio = float(self.param('bright_exit_ratio', 1.8))
        self.last_bright_log = -1e9
        self.create_subscription(
            CameraInfo, self.param('camera_info_topic', '/zed/zed_node/rgb/color/rect/camera_info'),
            self.on_info, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.param('image_topic', '/zed/zed_node/rgb/color/rect/image'),
            self.on_image, qos_profile_sensor_data)

    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def on_image(self, msg):
        if self.K is None:
            return
        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        bearing = bright_exit(gray, self.K[0, 0], self.K[0, 2], min_ratio=self.bright_ratio)
        self.pub_bright.publish(Float32(data=float('nan') if bearing is None else bearing))
        if bearing is not None and self.now_s() - self.last_bright_log > 2.0:
            self.last_bright_log = self.now_s()
            self.get_logger().info(f'bright exit at {math.degrees(bearing):+.0f} deg')
        corners, ids, rejected = self.detect(gray)
        if ids is None:
            # Dim tunnel / backlit tag: retry on a locally contrast-equalised image.
            corners, ids, rejected = self.detect(self.clahe.apply(gray))
        debug = [] if self.pub_debug.get_subscription_count() else None
        seen = set()
        out = []
        if ids is not None:
            for c, tag in zip(corners, ids.flatten()):
                tag = int(tag)
                zone = self.zone_of.get(tag)
                pts = c.reshape(4, 2).astype(np.float32)
                side = float(np.linalg.norm(pts[0] - pts[1]))
                d = None if debug is None else {'id': tag, 'zone': zone, 'side_px': round(side, 1),
                                                'corners': pts.round(1).tolist(), 'dist': None,
                                                'bearing': None, 'normal': None}
                if d is not None:
                    debug.append(d)
                if zone is None or side < self.min_side:
                    if d is not None:
                        d['status'] = 'unknown id' if zone is None else 'too small'
                    continue
                ok, rvec, tvec = cv2.solvePnP(self.obj, pts, self.K, self.D,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    if d is not None:
                        d['status'] = 'pose failed'
                    continue
                t = tvec.flatten()
                dist = float(np.linalg.norm(t))
                # Optical frame: x right, z forward -> bearing positive to the left.
                bearing = math.degrees(math.atan2(-t[0], t[2]))
                # Tag face normal in the vehicle frame (tags hang flat on course walls).
                n = cv2.Rodrigues(rvec)[0][:, 2]
                normal = math.degrees(math.atan2(-n[0], n[2]))
                if d is not None:
                    d.update(dist=round(dist, 2), bearing=round(bearing, 1), normal=round(normal, 1),
                             status='too far' if dist > self.max_range else
                             ('TRIGGER' if dist <= self.trigger_range else 'seen'))
                if dist > self.max_range:
                    continue
                out.append({'id': tag, 'zone': zone, 'distance': round(dist, 2),
                            'bearing_deg': round(bearing, 1), 'normal_deg': round(normal, 1)})
                if dist <= self.trigger_range:
                    seen.add(zone)
        for zone in list(self.hits):
            if zone not in seen:
                self.hits[zone] = 0
        now = self.now_s()
        for zone in seen:
            self.hits[zone] = self.hits.get(zone, 0) + 1
            if self.hits[zone] >= self.confirm and now - self.last_sent.get(zone, -1e9) > self.repeat_s:
                self.last_sent[zone] = now
                self.pub_zone.publish(String(data=zone))
                self.get_logger().info(f'marker -> zone {zone}')
        if out:
            self.pub_det.publish(String(data=dumps(out)))
        if debug is not None:
            self.pub_debug.publish(String(data=dumps({
                'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'w': gray.shape[1], 'h': gray.shape[0], 'tags': debug,
                'rejected': [r.reshape(4, 2).round(0).tolist() for r in (rejected or [])[:20]]})))


def main(args=None):
    run(MarkerDetector, args)
