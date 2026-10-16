#!/usr/bin/env python3
"""Standalone green-circle start signal (no tt_obstacle_course imports).

Publishes std_msgs/Int32 on `topic` (default /green_start) at 10 Hz: 0 until a green
CIRCLE (any shade, light to dark) is seen in the top part of the frame in
`confirm_frames` frames in a row, then 1 forever, until the node is killed.
Green things that are not circles (shirts, tape, leaves, bars) are rejected.

Every green blob it looks at, and why it was accepted or rejected, goes to
`<topic>/debug` (JSON); tt_dash draws it on the camera view.

  python3 scripts/green_start.py
  python3 scripts/green_start.py --ros-args -p image_topic:=/camera/image_raw -p topic:=/start
  ros2 topic echo /green_start

Parameters: image_topic, topic, min_radius_px (full resolution), top_frac (search only
above this fraction of the image height), min_circularity, min_fill, max_aspect,
min_solidity, confirm_frames, downscale, hsv_lower, hsv_upper.
"""

import json
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

# Any green from pale to dark, through teal (hue ~90-95): hue band, saturation clear of
# greys/whites, value above black.
GREEN_LOWER = [35, 70, 25]
GREEN_UPPER = [95, 255, 255]


def green_blobs(bgr, lower=GREEN_LOWER, upper=GREEN_UPPER, min_radius=4.5, top_frac=0.5,
                min_circularity=0.7, min_fill=0.7, max_aspect=1.3, min_solidity=0.88):
    """Every green blob big enough to judge, as dicts {u, v, r, box, ok, why}; circles
    that pass all tests first, biggest first. Only the part above top_frac of the frame
    is searched. A circle must be over min_radius (half its longer side), round
    (4*pi*A/P^2), fill its enclosing circle, be about as wide as tall (max_aspect 1.3
    lets a 5 x 4 px oval through) and have no dents (area / hull)."""
    roi = bgr[:max(1, int(top_frac * bgr.shape[0]))]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    # 3x3: clears 1-2 px specks (green wall texture, JPEG edges) but keeps a 4 px disc.
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    blobs = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        # Radius from the pixel box: half its longer side (blur/JPEG eat ~1 px of edge,
        # so a drawn radius-5 disc spans ~9 px -> 4.5).
        r = max(bw, bh) / 2.0
        if r <= min_radius - 1.5:
            continue
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        px_area = area + perim / 2.0 + 1.0
        (u, v), r_enc = cv2.minEnclosingCircle(c)
        circ = 4.0 * math.pi * px_area / (perim + 4.0) ** 2 if perim > 0 else 0.0
        # Share of the enclosing circle: a disc ~0.85+, a 5x4 oval ~0.75, a square ~0.64.
        fill = px_area / (math.pi * (r_enc + 0.5) ** 2)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull if hull > 0 else 0.0
        if r < min_radius:
            why = f'too small r={r:.1f}'
        elif aspect > max_aspect:
            why = f'not square {aspect:.2f}'
        elif circ < min_circularity:
            why = f'not round {circ:.2f}'
        elif fill < min_fill:
            why = f'not filled {fill:.2f}'
        elif solidity < min_solidity:
            why = f'dented {solidity:.2f}'
        else:
            why = f'circle r={r:.1f} round {circ:.2f}'
        if not why.startswith('circle') and r >= 8:
            # A disc joined to something thin (stick, tape): its largest inscribed circle.
            disc = inscribed_disc(c, min_radius)
            if disc is not None:
                u, v, r = disc
                why = f'circle r={r:.0f} (on a stick/edge)'
        blobs.append({'u': float(u), 'v': float(v), 'r': float(r), 'box': [x, y, bw, bh],
                      'ok': why.startswith('circle'), 'why': why})
    blobs.sort(key=lambda b: (not b['ok'], -b['r']))
    return blobs[:40]


def inscribed_disc(contour, min_radius, max_ring=0.2):
    """A disc joined to something thin (a stick, a strip of tape): the blob's largest
    inscribed circle, if it is at least min_radius and the blob barely reaches past it
    (green in the ring 1.05-1.3 r below max_ring; a square's corners fill ~45%)."""
    x, y, bw, bh = cv2.boundingRect(contour)
    blob = np.zeros((bh + 2, bw + 2), np.uint8)
    cv2.drawContours(blob, [contour - [x - 1, y - 1]], -1, 255, -1)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    _, r_in, _, (cu, cv_) = cv2.minMaxLoc(dist)
    if r_in < min_radius:
        return None
    yy, xx = np.mgrid[0:blob.shape[0], 0:blob.shape[1]]
    d = np.hypot(xx - cu, yy - cv_)
    ring = (d >= 1.05 * r_in) & (d <= 1.3 * r_in)
    if ring.sum() == 0 or (blob[ring] > 0).mean() > max_ring:
        return None
    return x - 1 + cu, y - 1 + cv_, r_in


class GreenStart(Node):

    def __init__(self):
        super().__init__('green_start')
        p = self.declare_parameter
        self.image_topic = p('image_topic', '/zed/zed_node/rgb/color/rect/image').value
        # Radius = half the longer side: must be over 4 px (a 5 x 4 px oval passes).
        self.min_radius = float(p('min_radius_px', 4.5).value)
        self.top_frac = float(p('top_frac', 0.5).value)
        self.min_circ = float(p('min_circularity', 0.7).value)
        self.min_fill = float(p('min_fill', 0.7).value)
        self.max_aspect = float(p('max_aspect', 1.3).value)
        self.min_solidity = float(p('min_solidity', 0.88).value)
        self.confirm = int(p('confirm_frames', 3).value)
        # Full resolution: a far start light is only a few px across (only the top part is searched).
        self.scale = float(p('downscale', 1.0).value)
        self.lower = [int(v) for v in p('hsv_lower', GREEN_LOWER).value]
        self.upper = [int(v) for v in p('hsv_upper', GREEN_UPPER).value]
        topic = p('topic', '/green_start').value
        self.value = 0
        self.hits = 0
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Int32, topic, 10)
        self.pub_debug = self.create_publisher(String, topic + '/debug', 2)
        self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_timer(0.1, lambda: self.pub.publish(Int32(data=self.value)))
        self.get_logger().info(f'watching {self.image_topic} for a green circle in the top '
                               f'{self.top_frac:.0%} of the frame; publishing 0')

    def on_image(self, msg):
        watched = self.pub_debug.get_subscription_count() > 0
        if self.value and not watched:
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        s = self.scale
        small = cv2.resize(bgr, None, fx=s, fy=s) if s != 1.0 else bgr
        blobs = green_blobs(small, self.lower, self.upper, self.min_radius * s, self.top_frac,
                            self.min_circ, self.min_fill, self.max_aspect, self.min_solidity)
        found = blobs[0] if blobs and blobs[0]['ok'] else None
        if not self.value:
            self.hits = self.hits + 1 if found else 0
            if self.hits >= self.confirm:
                self.value = 1
                self.pub.publish(Int32(data=1))
                self.get_logger().warn(f'GREEN circle at ({found["u"] / s:.0f}, {found["v"] / s:.0f}) '
                                       f'px r={found["r"] / s:.0f} -> publishing 1 until killed')
        if watched:
            out = [{'u': round(b['u'] / s), 'v': round(b['v'] / s), 'r': round(b['r'] / s),
                    'box': [round(t / s) for t in b['box']], 'ok': b['ok'], 'why': b['why']}
                   for b in blobs[:12]]
            self.pub_debug.publish(String(data=json.dumps({
                'w': bgr.shape[1], 'h': bgr.shape[0], 'top_frac': self.top_frac,
                'value': self.value, 'hits': self.hits, 'confirm': self.confirm, 'blobs': out})))


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
