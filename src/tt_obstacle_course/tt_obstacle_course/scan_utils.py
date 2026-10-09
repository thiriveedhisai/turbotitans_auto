"""LaserScan analysis in the robot frame (x forward, y left). No ROS imports.

Angles are rotated so 0 = vehicle forward, + = left, regardless of how the
lidar is mounted (see yaw_offset / flip).
"""

import math

import numpy as np

DEG = math.pi / 180.0


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class ScanView:

    def __init__(self, ranges, angle_min, angle_increment, range_min, range_max,
                 yaw_offset=0.0, flip=False):
        r = np.asarray(ranges, dtype=float)
        a = angle_min + np.arange(r.size) * angle_increment
        if flip:
            a = -a
        a = _wrap(a + yaw_offset)
        valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
        r = np.where(valid, r, np.inf)
        order = np.argsort(a)
        self.angles = a[order]
        self.ranges = r[order]

    @classmethod
    def from_msg(cls, msg, yaw_offset=0.0, flip=False, range_min=None):
        return cls(msg.ranges, msg.angle_min, msg.angle_increment,
                   msg.range_min if range_min is None else range_min,
                   msg.range_max, yaw_offset, flip)

    # ---------------------------------------------------------------- basics
    def sector(self, a0, a1):
        m = (self.angles >= a0) & (self.angles <= a1)
        return self.angles[m], self.ranges[m]

    def sector_min(self, a0, a1):
        _, r = self.sector(a0, a1)
        return float(r.min()) if r.size else math.inf

    def front_clearance(self, half_angle, ignore_half_angle=0.0):
        a, r = self.sector(-half_angle, half_angle)
        if ignore_half_angle > 0.0:
            r = r[np.abs(a) > ignore_half_angle]
        return float(r.min()) if r.size else math.inf

    def rear_clearance(self, half_angle):
        m = np.abs(self.angles) >= math.pi - half_angle
        r = self.ranges[m]
        return float(r.min()) if r.size else math.inf

    def corridor_clearance(self, half_width, forward=True, ignore_half_angle=0.0, min_x=0.0):
        """Distance to the nearest point inside the straight strip the vehicle
        would sweep (|y| <= half_width), ahead or behind. Side walls outside
        the strip are ignored, unlike an angular cone. Points closer than min_x
        (alongside the body, behind the bumper) are ignored: stopping can't help."""
        m = np.isfinite(self.ranges)
        if ignore_half_angle > 0.0:
            m &= np.abs(self.angles) > ignore_half_angle
        a, r = self.angles[m], self.ranges[m]
        x, y = r * np.cos(a), r * np.sin(a)
        ahead = (x > min_x) if forward else (x < -min_x)
        sel = (np.abs(y) <= half_width) & ahead
        return float(np.abs(x[sel]).min()) if sel.any() else math.inf

    def range_at(self, angle, half_window=3 * DEG):
        a, r = self.sector(angle - half_window, angle + half_window)
        r = r[np.isfinite(r)]
        return float(np.median(r)) if r.size else math.inf

    def points(self, a0=-math.pi, a1=math.pi):
        a, r = self.sector(a0, a1)
        m = np.isfinite(r)
        return r[m] * np.cos(a[m]), r[m] * np.sin(a[m])

    # ----------------------------------------------------------------- walls
    def side_distance(self, side, half_window=20 * DEG):
        """Perpendicular distance to the wall on ``side`` (+1 left, -1 right)."""
        a, r = self.sector(side * 90 * DEG - half_window, side * 90 * DEG + half_window)
        d = (r * np.abs(np.sin(a)))[np.isfinite(r)]
        if d.size < 3:
            return math.inf
        return float(np.percentile(d, 20))

    def far_side_wall(self, side, max_range=5.0):
        """Perpendicular distance to the wall on ``side`` behind any obstacles in
        front of it (farthest consistent returns, 60-120 deg). None if unseen."""
        lo, hi = sorted((side * 60 * DEG, side * 120 * DEG))
        a, r = self.sector(lo, hi)
        ok = r < max_range
        d = np.sort(r[ok] * np.abs(np.sin(a[ok])))
        return float(d[int(0.8 * (d.size - 1))]) if d.size >= 5 else None

    def wall_estimate(self, side, theta=45 * DEG, lookahead=0.0, max_range=3.0):
        """Two-beam wall fit. Returns (projected_distance, alpha) or None.

        alpha > 0 means the vehicle is heading away from that wall.
        """
        b = self.range_at(side * 90 * DEG)
        a = self.range_at(side * (90 * DEG - theta))
        if not (b < max_range and a < max_range / math.cos(theta)):
            return None
        alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
        dist = b * math.cos(alpha)
        return dist + lookahead * math.sin(alpha), alpha

    # --------------------------------------------------------------- openings
    def find_side_openings(self, side, min_width, depth_beyond,
                           a_from=15 * DEG, a_to=165 * DEG):
        """Openings in the wall on ``side`` that lead somewhere.

        A beam is 'far' if it reaches depth_beyond past the local wall line;
        closed notches (decoys) fail that test. Returns a list of dicts with
        x, y (gap centre, robot frame), yaw (direction through the gap),
        width, sorted nearest first.
        """
        m = (side * self.angles >= a_from) & (side * self.angles <= a_to)
        a = self.angles[m]
        r = self.ranges[m]
        if side < 0:
            a, r = a[::-1], r[::-1]
        return _openings(a, r, r * np.abs(np.sin(a)), min_width, depth_beyond)

    # ------------------------------------------------------------------ gates
    def find_gates(self, gate_width, tol, post_max_extent=0.15,
                   cluster_gap=0.10, max_range=3.5, half_fov=70 * DEG, isolation=0.25):
        """Pairs of thin, free-standing posts ``gate_width`` apart in front (hoop legs).

        A post must have no other returns within ``isolation`` m (apart from its
        partner), which rejects wall corners and bale ends.
        """
        a, r = self.sector(-half_fov, half_fov)
        m = np.isfinite(r) & (r <= max_range)
        a, r = a[m], r[m]
        if a.size == 0:
            return []
        xs, ys = r * np.cos(a), r * np.sin(a)
        allx, ally = self.points()
        clusters = []
        start = 0
        for k in range(1, a.size + 1):
            if k == a.size or math.hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1]) > cluster_gap:
                cx, cy = xs[start:k], ys[start:k]
                extent = math.hypot(cx[-1] - cx[0], cy[-1] - cy[0])
                if extent <= post_max_extent:
                    clusters.append((float(cx.mean()), float(cy.mean())))
                start = k

        def isolated(p, partner):
            d_p = np.hypot(allx - p[0], ally - p[1])
            d_q = np.hypot(allx - partner[0], ally - partner[1])
            near = (d_p > post_max_extent / 2 + 0.03) & (d_p < isolation) & (d_q > 0.08)
            return not near.any()

        gates = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                (x1, y1), (x2, y2) = clusters[i], clusters[j]
                d = math.hypot(x2 - x1, y2 - y1)
                if abs(d - gate_width) > tol:
                    continue
                if not (isolated(clusters[i], clusters[j]) and isolated(clusters[j], clusters[i])):
                    continue
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                nx, ny = -(y2 - y1), x2 - x1
                if nx * cx + ny * cy < 0:
                    nx, ny = -nx, -ny
                gates.append({'x': cx, 'y': cy, 'yaw': math.atan2(ny, nx), 'width': d})
        gates.sort(key=lambda g: math.hypot(g['x'], g['y']))
        return gates

    # ------------------------------------------------------------ gap follow
    def follow_gap(self, max_range, bubble_radius, free_range, half_fov=90 * DEG,
                   goal_angle=None, goal_gain=0.0, min_width=0.0):
        """Follow-the-gap. Returns (target_angle, target_distance) or None."""
        a, r = self.sector(-half_fov, half_fov)
        if a.size == 0:
            return None
        r = np.minimum(r, max_range)
        k = int(np.argmin(r))
        rc = float(r[k])
        if rc < max_range:
            half = math.asin(min(1.0, bubble_radius / max(rc, 1e-3)))
            r = np.where(np.abs(a - a[k]) <= half, 0.0, r)
        free = r >= free_range
        best = None
        i = 0
        while i < a.size:
            if not free[i]:
                i += 1
                continue
            j = i
            while j + 1 < a.size and free[j + 1]:
                j += 1
            depth = float(r[i:j + 1].mean())
            width = 2.0 * min(r[i], r[j]) * math.sin((a[j] - a[i]) / 2.0)
            if width >= min_width:
                centre = float((a[i] + a[j]) / 2.0)
                score = depth + width
                if goal_angle is not None:
                    score -= goal_gain * abs(centre - goal_angle)
                if best is None or score > best[0]:
                    best = (score, centre, depth)
            i = j + 1
        if best is None:
            return None
        return best[1], best[2]


def pursuit_curvature(target_angle, target_distance):
    """Pure-pursuit curvature to a point at (angle, distance) in robot frame."""
    return 2.0 * math.sin(target_angle) / max(target_distance, 0.3)


def _openings(a, r, dist, min_width, depth_beyond):
    """Runs of beams reaching depth_beyond past the wall line (``dist`` is each
    beam's distance measured perpendicular to that wall); closed notches fail."""
    if a.size < 5:
        return []
    finite = np.isfinite(dist)
    if finite.sum() < 5:
        return []
    wall = float(np.percentile(dist[finite], 30))
    far = ~finite | (dist > wall + depth_beyond)
    out = []
    i = 0
    n = a.size
    while i < n:
        if not far[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and far[j + 1]:
            j += 1
        if i > 0 and j < n - 1 and finite[i - 1] and finite[j + 1]:
            p1 = np.array([r[i - 1] * math.cos(a[i - 1]), r[i - 1] * math.sin(a[i - 1])])
            p2 = np.array([r[j + 1] * math.cos(a[j + 1]), r[j + 1] * math.sin(a[j + 1])])
            width = float(np.linalg.norm(p2 - p1))
            if width >= min_width:
                c = (p1 + p2) / 2.0
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                if np.dot(normal, c) < 0:
                    normal = -normal
                out.append({'x': float(c[0]), 'y': float(c[1]),
                            'yaw': math.atan2(normal[1], normal[0]), 'width': width})
        i = j + 1
    out.sort(key=lambda g: math.hypot(g['x'], g['y']))
    return out
