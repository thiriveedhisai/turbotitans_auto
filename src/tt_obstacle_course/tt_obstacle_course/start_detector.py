"""Start detector: waits for the pinwheel to turn green, then latches /race_start.

A detection is a green blob that is round enough and within a depth window.
With require_transition, green must appear after a period without it, so a
green object already in view at boot does not trigger a false start.
Manual start: ros2 topic pub --once /race_start std_msgs/msg/Bool "{data: true}"
"""

import math

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from tt_obstacle_course.ros_common import LATCHED, RELIABLE_SUB, TTNode, run


class StartDetector(TTNode):

    def __init__(self):
        super().__init__('start_detector')
        self.lower = np.array(self.param('hsv_lower', [40, 80, 60]), dtype=np.uint8)
        self.upper = np.array(self.param('hsv_upper', [85, 255, 255]), dtype=np.uint8)
        self.min_area = float(self.param('min_area_px', 400))
        self.min_circ = float(self.param('min_circularity', 0.6))
        self.dmin = float(self.param('depth_min', 0.3))
        self.dmax = float(self.param('depth_max', 4.0))
        self.confirm = int(self.param('confirm_frames', 6))
        self.require_transition = bool(self.param('require_transition', True))
        self.absent_needed = int(self.param('absent_frames', 5))
        self.scale = float(self.param('downscale', 0.5))

        self.bridge = CvBridge()
        self.depth = None
        self.absent = 0
        self.armed = not self.require_transition
        self.hits = 0
        self.started = False

        self.pub = self.create_publisher(Bool, '/race_start', LATCHED)
        self.create_subscription(Bool, '/race_start', self.on_start, RELIABLE_SUB)
        self.create_subscription(
            Image, self.param('depth_topic', '/zed/zed_node/depth/depth_registered'),
            self.on_depth, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.param('image_topic', '/zed/zed_node/rgb/color/rect/image'),
            self.on_image, qos_profile_sensor_data)
        self.get_logger().info('Waiting for green start signal...')

    def on_start(self, msg):
        if msg.data:
            self.started = True

    def on_depth(self, msg):
        if not self.started:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def depth_at(self, u, v, img_w, img_h):
        if self.depth is None:
            return None
        dh, dw = self.depth.shape[:2]
        x = int(u * dw / img_w)
        y = int(v * dh / img_h)
        patch = self.depth[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        return float(np.median(vals)) if vals.size else None

    def find_green(self, bgr):
        small = cv2.resize(bgr, None, fx=self.scale, fy=self.scale) if self.scale != 1.0 else bgr
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = small.shape[:2]
        min_area = self.min_area * self.scale * self.scale
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < min_area:
                break
            perim = cv2.arcLength(c, True)
            if perim <= 0 or 4.0 * math.pi * area / (perim * perim) < self.min_circ:
                continue
            m = cv2.moments(c)
            d = self.depth_at(m['m10'] / m['m00'], m['m01'] / m['m00'], w, h)
            if d is not None and self.dmin <= d <= self.dmax:
                return d
        return None

    def on_image(self, msg):
        if self.started:
            return
        d = self.find_green(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
        if d is None:
            self.hits = 0
            self.absent += 1
            if not self.armed and self.absent >= self.absent_needed:
                self.armed = True
                self.get_logger().info('Armed: no green in view, waiting for it to appear.')
            return
        self.absent = 0
        if not self.armed:
            return
        self.hits += 1
        if self.hits >= self.confirm:
            self.started = True
            self.pub.publish(Bool(data=True))
            self.get_logger().info(f'GREEN at {d:.2f} m -> race start!')


def main(args=None):
    run(StartDetector, args)
