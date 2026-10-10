"""Camera depth checks for lidar detections (no ROS imports).

Body frame: x forward, y left, z up, origin at the camera; heights are above ground.
"""

import math

import numpy as np


class DepthView:
    """A ZED registered depth image (metres, invalid = nan/inf/0) with pinhole intrinsics."""

    def __init__(self, depth, K, cam_height):
        self.d = depth
        self.fx, self.fy, self.cx, self.cy = K[0][0], K[1][1], K[0][2], K[1][2]
        self.cam_h = cam_height
        self.h, self.w = depth.shape

    def pixel(self, x, y, height):
        """Body point (x fwd, y left, height above ground) -> (u, v) or None if not in view."""
        if x < 0.2:
            return None
        u = self.cx - self.fx * y / x
        v = self.cy - self.fy * (height - self.cam_h) / x
        if not (0 <= u < self.w and 0 <= v < self.h):
            return None
        return int(u), int(v)

    def sample(self, x, y, height, half=2):
        """Valid depths in a small window around the projected point (empty if not in view)."""
        p = self.pixel(x, y, height)
        if p is None:
            return np.empty(0)
        u, v = p
        win = self.d[max(0, v - half):v + half + 1, max(0, u - half):u + half + 1].ravel()
        return win[np.isfinite(win) & (win > 0.05)]

    def hoop_frame(self, legs, top_min=0.55, top_max=1.4, body_h=(0.10, 0.35), tol=0.25, min_range=1.1):
        """Is the leg pair (body-frame (x, y) each) a hoop seen in depth?

        Returns True (legs + a top spanning the opening at the gate's distance + see-through
        opening), False (in view but not a hoop: e.g. a wall end, bale corner or notch), or
        None (top not in the camera's view, e.g. too close: can't tell)."""
        (x1, y1), (x2, y2) = legs
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Closer than this the top is above the field of view (camera ~0.2 m up, +-30 deg).
        if cx < min_range:
            return None
        top_seen, top_in_view = False, False
        for h in np.arange(top_min, top_max, 0.05):
            for f in (-0.4, -0.2, 0.0, 0.2, 0.4):
                px, py = cx + f * (x2 - x1), cy + f * (y2 - y1)
                d = self.sample(px, py, h)
                if self.pixel(px, py, h) is not None:
                    top_in_view = True
                if d.size and np.any(np.abs(d - px) < tol):
                    top_seen = True
        if not top_in_view:
            return None
        if not top_seen:
            return False
        through = []
        for h in np.arange(body_h[0], body_h[1], 0.05):
            p = self.pixel(cx, cy, h)
            if p is None:
                continue
            u, v = p
            win = self.d[max(0, v - 2):v + 3, max(0, u - 2):u + 3].ravel()
            ok = np.isfinite(win) & (win > 0.05)
            # Beyond the gate, or no return at all (far / sky), counts as open.
            through.append(bool(np.mean(~ok | (win > cx + 0.4)) > 0.6))
        if not through or sum(through) < 0.6 * len(through):
            return False
        legs_seen = 0
        for lx, ly in legs:
            for h in (0.25, 0.5):
                d = self.sample(lx, ly, h)
                legs_seen += int(d.size > 0 and bool(np.any(np.abs(d - lx) < 0.2)))
        return legs_seen >= 1


def leg_points(gate):
    """The two leg positions (lidar frame) of a find_gates() result."""
    half = gate['width'] / 2
    nx, ny = math.cos(gate['yaw']), math.sin(gate['yaw'])
    # Legs lie across the passing direction.
    return ((gate['x'] - ny * half, gate['y'] + nx * half), (gate['x'] + ny * half, gate['y'] - nx * half))
