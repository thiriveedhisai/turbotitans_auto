#!/usr/bin/env python3
"""Standalone green-circle start signal (no tt_obstacle_course imports).

Publishes std_msgs/Int32 on `topic` (default /green_start) at 10 Hz: 0 until a green
circle (any shade, light to dark) is seen in `confirm_frames` frames in a row, then 1
forever, until the node is killed.

  python3 scripts/green_start.py
  python3 scripts/green_start.py --ros-args -p image_topic:=/camera/image_raw -p topic:=/start
  ros2 topic echo /green_start

Parameters: image_topic, topic, min_area_px (at full resolution), min_circularity,
min_disc_fraction, confirm_frames, downscale, hsv_lower, hsv_upper.
"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

# Any green from pale to dark: hue band only, saturation above greys/whites, value above black.
GREEN_LOWER = [35, 60, 25]
GREEN_UPPER = [90, 255, 255]


def green_circles(bgr, lower=GREEN_LOWER, upper=GREEN_UPPER, min_area=400.0,
                  min_circularity=0.6, min_disc_fraction=0.45):
    """Centres (u, v) of green round blobs, largest first. A blob counts if it is round
    (4*pi*A/P^2), or its largest inscribed circle covers min_disc_fraction of it (a disc
    on a green stick, a slightly oval or ragged print)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area:
            break
        perim = cv2.arcLength(c, True)
        if perim > 0 and 4.0 * math.pi * area / (perim * perim) >= min_circularity:
            m = cv2.moments(c)
            out.append((m['m10'] / m['m00'], m['m01'] / m['m00']))
            continue
        x, y, w, h = cv2.boundingRect(c)
        blob = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(blob, [c - [x - 1, y - 1]], -1, 255, -1)
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        _, r_in, _, (cu, cv_) = cv2.minMaxLoc(dist)
        disc = math.pi * r_in * r_in
        if disc >= min_disc_fraction * area and disc >= min_area / 2:
            out.append((x - 1 + cu, y - 1 + cv_))
    return out


class GreenStart(Node):

    def __init__(self):
        super().__init__('green_start')
        p = self.declare_parameter
        self.image_topic = p('image_topic', '/zed/zed_node/rgb/color/rect/image').value
        self.min_area = float(p('min_area_px', 400.0).value)
        self.min_circ = float(p('min_circularity', 0.6).value)
        self.min_disc = float(p('min_disc_fraction', 0.45).value)
        self.confirm = int(p('confirm_frames', 3).value)
        self.scale = float(p('downscale', 0.5).value)
        self.lower = [int(v) for v in p('hsv_lower', GREEN_LOWER).value]
        self.upper = [int(v) for v in p('hsv_upper', GREEN_UPPER).value]
        self.value = 0
        self.hits = 0
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Int32, p('topic', '/green_start').value, 10)
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image,
                                            qos_profile_sensor_data)
        self.create_timer(0.1, lambda: self.pub.publish(Int32(data=self.value)))
        self.get_logger().info(f'watching {self.image_topic} for a green circle; publishing 0')

    def on_image(self, msg):
        if self.value:
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale)
        found = green_circles(bgr, self.lower, self.upper, self.min_area * self.scale ** 2,
                              self.min_circ, self.min_disc)
        self.hits = self.hits + 1 if found else 0
        if self.hits >= self.confirm:
            self.value = 1
            self.pub.publish(Int32(data=1))
            u, v = found[0]
            self.get_logger().warn(f'GREEN circle at ({u / self.scale:.0f}, {v / self.scale:.0f}) px '
                                   f'-> publishing 1 until killed')
            # Latched: no need to keep decoding images.
            self.destroy_subscription(self.sub)


def main():
    rclpy.init()
    node = GreenStart()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
