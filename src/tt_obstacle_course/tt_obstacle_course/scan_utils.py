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
                 yaw_offset=0.0, flip=False, self_mask=()):
        r = np.asarray(ranges, dtype=float)
        a = angle_min + np.arange(r.size) * angle_increment
        if flip:
            a = -a
        a = _wrap(a + yaw_offset)
        valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
        r = np.where(valid, r, np.inf)
        # Parts of our own truck (the flag pole near the rear-right corner): drop returns in
        # a narrow direction only, [bearing_deg, half_width_deg, max_range] in the vehicle
        # frame. A circle around the lidar would also hide real obstacles close alongside.
        for bearing, half, max_r in self_mask:
            off = np.abs(_wrap(a - math.radians(bearing)))
            r = np.where((off <= math.radians(half)) & (r <= max_r), np.inf, r)
        order = np.argsort(a)
        self.angles = a[order]
        self.ranges = r[order]

    @classmethod
    def from_msg(cls, msg, yaw_offset=0.0, flip=False, range_min=None, self_mask=()):
        return cls(msg.ranges, msg.angle_min, msg.angle_increment,
                   msg.range_min if range_min is None else range_min,
                   msg.range_max, yaw_offset, flip, self_mask)

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

    def corridor_clearance(self, half_width, forward=True, ignore_half_angle=0.0, min_x=0.0,
                           curvature=0.0, axle_back=0.0, body_half=None, taper=0.15):
        """Distance to the nearest point inside the straight strip the vehicle
        would sweep (|y| <= half_width), ahead or behind. Side walls outside
        the strip are ignored, unlike an angular cone. Points closer than min_x
        (alongside the body, behind the bumper) are ignored: stopping can't help.

        body_half: the strip's safety margin (half_width - body_half) grows from 0 at
        the bumper to full over `taper` m. A lane wall 5 cm beside the bumper isn't in
        the path (the steered band covers the drift); a full margin there stopped the
        truck in every tight lane once the bumper was measured at 0.152 m, not 0.20.

        curvature != 0: check the arc band swept at that curvature instead
        (rotation centre on the rear-axle line, axle_back metres behind the
        lidar) and return the arc length to the nearest point in it. A wall
        alongside the truck while it steers away no longer reads as blocking."""
        m = np.isfinite(self.ranges)
        if ignore_half_angle > 0.0:
            m &= np.abs(self.angles) > ignore_half_angle
        a, r = self.angles[m], self.ranges[m]
        x, y = r * np.cos(a), r * np.sin(a)

        def band(s):
            if body_half is None or taper <= 0.0:
                return half_width
            return body_half + (half_width - body_half) * np.clip((s - min_x) / taper, 0.0, 1.0)
        if abs(curvature) > 1e-3:
            side = 1.0 if curvature > 0 else -1.0
            cx, cy = -axle_back, side / abs(curvature)
            r_l = math.hypot(cx, cy)
            phi0 = math.atan2(-cy, -cx)
            dphi = side * _wrap(np.arctan2(y - cy, x - cx) - phi0)
            if not forward:
                dphi = -dphi
            arc = r_l * dphi
            sel = (np.abs(np.hypot(x - cx, y - cy) - r_l) <= band(arc)) & (arc > min_x) & (dphi < math.pi / 2)
            return float(arc[sel].min()) if sel.any() else math.inf
        ahead = (x > min_x) if forward else (x < -min_x)
        sel = (np.abs(y) <= band(np.abs(x))) & ahead
        return float(np.abs(x[sel]).min()) if sel.any() else math.inf

    def range_at(self, angle, half_window=3 * DEG):
        a, r = self.sector(angle - half_window, angle + half_window)
        r = r[np.isfinite(r)]
        return float(np.median(r)) if r.size else math.inf

    def points(self, a0=-math.pi, a1=math.pi):
        a, r = self.sector(a0, a1)
        m = np.isfinite(r)
        return r[m] * np.cos(a[m]), r[m] * np.sin(a[m])

    def object_mask(self, max_size=0.45, min_jump=0.25, link=0.08):
        """Returns that belong to small free-standing objects (buckets, posts): a run of
        neighbouring returns at most max_size across with the background at least
        min_jump farther (or nothing) on both sides. Walls are long or end on one side only."""
        r, a = self.ranges, self.angles
        fin = np.isfinite(r)
        x = np.where(fin, r * np.cos(a), np.nan)
        y = np.where(fin, r * np.sin(a), np.nan)
        mask = np.zeros(r.size, dtype=bool)
        step = float(np.median(np.diff(a))) if a.size > 1 else DEG
        i, n = 0, r.size
        while i < n:
            if not fin[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and fin[j + 1] and math.hypot(x[j + 1] - x[j], y[j + 1] - y[j]) \
                    <= max(link, 2.5 * r[j] * step):
                j += 1
            near = float(r[i:j + 1].min())
            before = r[i - 1] if i > 0 else math.inf
            after = r[j + 1] if j + 1 < n else math.inf
            if (math.hypot(x[j] - x[i], y[j] - y[i]) <= max_size
                    and before >= near + min_jump and after >= near + min_jump):
                mask[i:j + 1] = True
            i = j + 1
        return mask

    def without_objects(self, **kw):
        """Copy with buckets/posts removed, for wall measurements only (the planner
        must still see them)."""
        out = object.__new__(ScanView)
        out.angles = self.angles
        out.ranges = np.where(self.object_mask(**kw), np.inf, self.ranges)
        return out

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

    # --------------------------------------------------------------- openings
    def walls_end(self, max_lat=1.5, min_lat=0.15, max_gap=0.35):
        """Distance ahead where the nearer of the two side walls stops (None if a
        side has no wall alongside). Each wall is followed forward from abeam
        while consecutive returns are < max_gap apart."""
        xs, ys = self.points()
        ends = []
        for side in (1, -1):
            m = (side * ys >= min_lat) & (side * ys <= max_lat) & (xs > -0.3)
            x = np.sort(xs[m])
            if x.size < 3 or x[0] > 0.3:
                return None
            jumps = np.nonzero(np.diff(x) > max_gap)[0]
            ends.append(float(x[jumps[0]] if jumps.size else x[-1]))
        return min(ends)

    def lane_walls(self, max_half_width=1.0, x_min=-0.25, x_max=1.6, bin_m=0.1, min_bins=4):
        """Fit each lidar wall: {+1: (y0, slope, n, curv) | None, -1: ...}, with
        y = y0 + slope * x + curv * x^2 in the robot frame (curv only when enough wall
        is seen to tell a bend from noise, else 0).

        Per x-bin only the point nearest the robot counts, so things behind the wall
        (or a bale's far face) are ignored; x_max should stop short of a wall ahead."""
        xs, ys = self.points()
        out = {}
        for side in (1, -1):
            m = (xs >= x_min) & (xs <= x_max) & (side * ys > 0.05) & (side * ys < max_half_width)
            if m.sum() < min_bins:
                out[side] = None
                continue
            bx, by = xs[m], side * ys[m]
            idx = np.floor((bx - x_min) / bin_m).astype(int)
            near = {}
            for i, x, y in zip(idx, bx, by):
                if i not in near or y < near[i][1]:
                    near[i] = (x, y)
            if len(near) < min_bins:
                out[side] = None
                continue
            px = np.array([p[0] for p in near.values()])
            py = np.array([p[1] for p in near.values()])
            fit = None
            for _ in range(2):
                if px.size < min_bins:
                    break
                slope, y0 = np.polyfit(px, py, 1)
                res = np.abs(py - (y0 + slope * px))
                fit = (y0, slope)
                keep = res < max(0.06, 2.5 * float(np.median(res)))
                if keep.all():
                    break
                px, py = px[keep], py[keep]
            if fit is None:
                out[side] = None
                continue
            curv = 0.0
            if px.size >= 8 and np.ptp(px) >= 0.8:
                c2, c1, c0 = np.polyfit(px, py, 2)
                if abs(c2) <= 0.8:
                    fit, curv = (c0, c1), c2
            out[side] = (side * fit[0], side * fit[1], int(px.size), side * curv)
        return out

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
        found = _openings(a, r, r * np.abs(np.sin(a)), min_width, depth_beyond)
        # Seen obliquely, the two edge returns sit at different depths (wall end vs far
        # jamb) and skew the edge-to-edge normal by 10-20 deg. The opening is square to
        # the wall it is cut into, so take the axis from the fitted wall when there is one.
        wall = self.lane_walls(max_half_width=2.0, x_min=-1.5, x_max=1.5).get(side)
        if wall is not None and wall[2] >= 6 and abs(wall[1]) < 0.7:
            axis = math.atan(wall[1]) + side * math.pi / 2
            for g in found:
                if abs(math.atan2(math.sin(g['yaw'] - axis), math.cos(g['yaw'] - axis))) < 35 * DEG:
                    g['yaw'] = axis
        return found

    def wall_line(self, side, max_dist=3.5, x_min=-1.0, x_max=4.0, tol=0.05, max_slope=0.7,
                  iters=120, min_span=0.8):
        """The dominant straight wall on ``side``: RANSAC over that side's returns.
        Returns (y0, slope, inliers) in the robot frame (signed y = y0 + slope * x) or None.
        The longest straight run wins, so decoy legs (steep), posts and buckets (round)
        and short stubs don't drag it the way a nearest-point fit does."""
        xs, ys = self.points()
        m = (xs >= x_min) & (xs <= x_max) & (side * ys > 0.15) & (side * ys < max_dist)
        px, py = xs[m], ys[m]
        n = px.size
        if n < 8:
            return None
        rng = np.random.default_rng(0)
        best, best_in = None, None
        pairs = rng.integers(0, n, size=(iters, 2))
        for i, j in pairs:
            dx = px[j] - px[i]
            if abs(dx) < 0.3:
                continue
            s = (py[j] - py[i]) / dx
            if abs(s) > max_slope:
                continue
            c = py[i] - s * px[i]
            inl = np.abs(py - (c + s * px)) / math.hypot(1.0, s) < tol
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = (c, s), inl
        if best is None or best_in.sum() < 8 or np.ptp(px[best_in]) < min_span:
            return None
        s, c = np.polyfit(px[best_in], py[best_in], 1)
        return float(c), float(s), int(best_in.sum())

    def flank_axis(self, cx, cy, cross, inner=0.4, outer=1.4, depth=1.0, min_pts=6,
                   min_elong=4.0):
        """Direction (rad, robot frame) of the bales flanking a hoop whose leg-pair centre
        is (cx, cy) and whose crossing direction is about `cross`: returns beyond each leg,
        along the leg line, fitted by PCA. The bales run along the lane, so a real hoop's
        crossing is parallel to them, whatever odometry heading says. None if neither side
        shows an elongated run of returns."""
        xs, ys = self.points()
        ux, uy = -math.sin(cross), math.cos(cross)     # along the leg line
        wx, wy = math.cos(cross), math.sin(cross)      # through the hoop
        dx, dy = xs - cx, ys - cy
        a = dx * ux + dy * uy
        b = dx * wx + dy * wy
        angles, weights = [], []
        for side in (1, -1):
            m = (side * a > inner) & (side * a < outer) & (np.abs(b) < depth)
            if m.sum() < min_pts:
                continue
            pts = np.stack([xs[m], ys[m]], axis=1)
            evals, evecs = np.linalg.eigh(np.cov(pts.T))
            if evals[1] < min_elong * max(evals[0], 1e-6):
                continue
            v = evecs[:, 1]
            angles.append(math.atan2(v[1], v[0]))
            weights.append(float(m.sum()))
        if not angles:
            return None
        # Average undirected angles (mod pi), then pick the sense nearest `cross`.
        s = sum(w * math.sin(2 * t) for t, w in zip(angles, weights))
        c = sum(w * math.cos(2 * t) for t, w in zip(angles, weights))
        ang = 0.5 * math.atan2(s, c)
        if abs(_wrap(ang - cross)) > math.pi / 2:
            ang = _wrap(ang + math.pi)
        return ang

    def find_wall_breaks(self, side, min_width, depth_beyond, max_wall=3.0, u_max=4.0,
                         on_wall=0.12, min_beams=2, max_width=1.4, support=0.5):
        """Openings in the wall on ``side`` measured along that wall: fit the wall line
        (wall_line), find stretches of it with no returns on it, and keep those that beams
        pass through to depth_beyond past it. Both ends are the opening's own jambs, so
        the centre is exact even seen obliquely (the edge-run method ended runs on buckets
        beyond a doorway, reading 3 m wide). Closed notches and things standing in front
        of the wall have no such beams.
        Returns dicts like find_side_openings (robot frame), nearest first."""
        line = self.wall_line(side, max_dist=max_wall, x_min=-0.6, x_max=u_max)
        if line is None:
            return []
        y0, slope = line[0], line[1]
        norm = math.hypot(1.0, slope)
        ux, uy = 1.0 / norm, slope / norm          # along the wall
        nx, ny = -uy * side, ux * side             # away from the robot, through the wall
        if nx * 0.0 + ny * y0 < 0:
            nx, ny = -nx, -ny
        m = (side * self.angles > 0.0) & (side * self.angles < math.pi)
        a, r = self.angles[m], self.ranges[m]
        fin = np.isfinite(r)
        px, py = r[fin] * np.cos(a[fin]), r[fin] * np.sin(a[fin])
        u = px * ux + (py - y0) * uy
        v = px * nx + (py - y0) * ny
        on = np.sort(u[(np.abs(v) <= on_wall) & (u > -0.6) & (u < u_max)])
        if on.size < 6:
            return []
        real = on
        # Where each beam crosses the wall line, and how far past it it reaches.
        dx, dy = np.cos(a), np.sin(a)
        denom = dx * nx + dy * ny
        t = np.where(np.abs(denom) > 1e-3, (0.0 * nx + y0 * ny) / np.where(denom == 0, 1, denom), -1)
        cross_u = t * (dx * ux + dy * uy) + (0.0 - 0.0) - y0 * uy
        beyond = np.where(np.isfinite(r), r - t, np.inf) * np.abs(denom)
        through = (t > 0) & (beyond >= depth_beyond)
        # A beam stopped short of the wall (something standing in front of it) says
        # nothing about the wall there: its shadow must not merge with a real opening.
        shadow = cross_u[(t > 0) & np.isfinite(r) & (beyond < -on_wall)]
        on = np.sort(np.concatenate([on, shadow[(shadow > -0.6) & (shadow < u_max)]]))
        out = []
        for u1, u2 in zip(on[:-1], on[1:]):
            if not min_width <= u2 - u1 <= max_width:
                continue
            # Real wall on both sides (a fit tilted onto something in front of the wall
            # crosses the wall at one point only).
            if (np.count_nonzero((real >= u1 - support) & (real <= u1)) < 3
                    or np.count_nonzero((real >= u2) & (real <= u2 + support)) < 3):
                continue
            if np.count_nonzero(through & (cross_u > u1) & (cross_u < u2)) < min_beams:
                continue
            c = (u1 + u2) / 2.0
            cx, cy = c * ux, y0 + c * uy
            out.append({'x': float(cx), 'y': float(cy), 'yaw': math.atan2(ny, nx),
                        'width': float(u2 - u1)})
        out.sort(key=lambda g: math.hypot(g['x'], g['y']))
        return out

    def find_front_openings(self, min_width, depth_beyond, half_angle=60 * DEG):
        """Openings in the wall ahead (roughly square to the heading) that lead somewhere."""
        a, r = self.sector(-half_angle, half_angle)
        return _openings(a, r, r * np.abs(np.cos(a)), min_width, depth_beyond)

    # ------------------------------------------------------------------ gates
    def find_gates(self, gate_width, tol, post_max_extent=0.15,
                   cluster_gap=0.10, max_range=3.5, half_fov=70 * DEG, isolation=0.25,
                   max_yaw=60 * DEG, post_jump=0.04):
        """Pairs of thin, free-standing posts ``gate_width`` apart in front (hoop legs),
        open between them and passable roughly forward (|yaw| <= max_yaw).

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
        # A leg a few cm in front of a wall (hoop 3 vs the U-turn bale wall) is closer
        # than cluster_gap and merges into it; find it as a short run stepping out of the
        # wall on both sides (a convex wall corner has no step either side).
        dips = []
        for s, e in _range_dips(r, post_jump):
            cx, cy = xs[s:e + 1], ys[s:e + 1]
            if math.hypot(cx[-1] - cx[0], cy[-1] - cy[0]) <= post_max_extent:
                dips.append((float(cx.mean()), float(cy.mean())))
        # Sparse returns from a wall seen at a grazing angle look like posts: drop
        # anything lying on a fitted side wall. Only near-parallel fits are walls; a steep
        # fit (e.g. across a U-turn bowl) is usually threaded through the legs themselves.
        # Dips already stand out of the wall, so they are kept.
        walls = [w for w in self.lane_walls(max_half_width=1.2, x_max=max_range).values()
                 if w and abs(w[1]) <= 0.6]
        clusters = [(px, py) for px, py in clusters
                    if all(abs(py - (w[0] + w[1] * px + w[3] * px * px)) > 0.1 for w in walls)]
        clusters += [p for p in dips
                     if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.05 for q in clusters)]

        def isolated(p, partner):
            d_p = np.hypot(allx - p[0], ally - p[1])
            d_q = np.hypot(allx - partner[0], ally - partner[1])
            near = (d_p > post_max_extent / 2 + 0.03) & (d_p < isolation) & (d_q > 0.08)
            # A wall just outside the outer leg (hoop 3 sits ~0.17 m off the U-turn wall)
            # is beyond the leg, away from its partner; corners/bale ends are not.
            ux, uy = p[0] - partner[0], p[1] - partner[1]
            norm = math.hypot(ux, uy) or 1.0
            outward = ((allx - p[0]) * ux + (ally - p[1]) * uy) / norm > 0.04
            return not (near & ~outward).any()

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
                # A hoop is open between its legs; two returns on a wall are not.
                if (np.hypot(allx - cx, ally - cy) < min(0.2, d / 3)).any():
                    continue
                nx, ny = -(y2 - y1), x2 - x1
                if nx * cx + ny * cy < 0:
                    nx, ny = -nx, -ny
                yaw = math.atan2(ny, nx)
                if abs(yaw) > max_yaw:
                    continue
                gates.append({'x': cx, 'y': cy, 'yaw': yaw, 'width': d})
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


def scan_shift(prev, cur, dyaw=0.0, max_shift=0.6, step=0.01, tol=0.03, max_range=6.0):
    """Forward travel between two scans (robot frame, ranges/angles arrays each), by a
    1-D match: rotate `prev` by the gyro's dyaw, slide it back by s and count points of
    `cur` that land within tol. Returns the best s, or None when the match is not sharp
    (featureless corridor, or nothing in range) - an honest 'don't know'."""
    (a0, r0), (a1, r1) = prev, cur
    m0 = np.isfinite(r0) & (r0 > 0.2) & (r0 < max_range)
    m1 = np.isfinite(r1) & (r1 > 0.2) & (r1 < max_range)
    if m0.sum() < 30 or m1.sum() < 30:
        return None

    def oblique(pts):
        # Surfaces along the travel direction (lane walls) match at every shift; keep
        # only points whose local surface (from scan neighbours) crosses it.
        t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        n = np.hypot(t[:, 0], t[:, 1])
        return pts[(n < 0.3) & (np.abs(t[:, 0]) < 0.75 * np.maximum(n, 1e-6))]
    p = oblique(np.stack([r0[m0] * np.cos(a0[m0] - dyaw), r0[m0] * np.sin(a0[m0] - dyaw)], axis=1))
    q = np.stack([r1[m1] * np.cos(a1[m1]), r1[m1] * np.sin(a1[m1])], axis=1)
    if len(p) < 15:
        return None
    p = p[::max(1, len(p) // 200)]
    q = q[::max(1, len(q) // 300)]
    shifts = np.arange(-max_shift, max_shift + step / 2, step)
    # (shifts, p, q) squared distances; a point of prev counts if any point of cur is within tol.
    dx = (p[None, :, 0] - shifts[:, None])[:, :, None] - q[None, None, :, 0]
    dy = p[None, :, None, 1] - q[None, None, :, 1]
    scores = ((dx * dx + dy * dy) <= tol * tol).any(axis=2).sum(axis=1)
    best = int(np.argmax(scores))
    # Sharp: clearly better than shifts 10+ cm away (a corridor scores flat along it).
    far = np.abs(shifts - shifts[best]) >= 0.10
    if not far.any() or scores[best] < 15 or scores[best] < 1.25 * scores[far].max():
        return None
    return float(shifts[best])


def _range_dips(r, jump, max_len=6):
    """(start, end) index runs at least `jump` nearer than the beams on both sides."""
    out = []
    n = r.size
    for s in range(1, n - 1):
        if not r[s - 1] - r[s] > jump:
            continue
        for e in range(s, min(n - 1, s + max_len)):
            if r[e + 1] - r[e] > jump:
                if r[s:e + 1].max() - r[s:e + 1].min() < jump:
                    out.append((s, e))
                break
    return out


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
