"""Lidar map of the course: build it from one drive, then localize on it in later runs.

  tt_map build ~/runs/bag_<name> -o course       map from a tt_record bag -> ~/maps/course/
  tt_map live -o course                          map while driving (Ctrl-C saves); shows in tt_dash
  tt_map update ~/maps/course ~/runs/bag_<heat> --zones buckets wide hoops -o course_heat1
                                                 re-map only those zones from a later run
  tt_map show ~/maps/course                      print info, redraw map.png
  tt_map eval ~/maps/course ~/runs/bag_<name>    localize through a bag, report the match quality
                                                 (and the error, if the bag has --truth)

Mapping: each level lidar scan is matched against the map built so far (correlative
scan matching seeded by odometry), then added to it. Scans while the lidar is tilted
(ramps, bank, helix: /terrain/lidar_ok false) are skipped. The map frame is the odometry
frame of the mapping drive, so a run started from the same start box needs no setup.

Localizing (map_localizer node): the same matcher against the saved map. Points with no
map wall near them (moved buckets/hoops, extra bales, a shifted entrance) simply don't
score, so a partly changed course still matches on the walls that stayed. Along a
featureless lane the along-lane position is left to odometry. Outputs the map pose, a
quality (fraction of scan points on a map wall) and the zone the mapping drive was in there.
Files: map.npz (grid + labelled trajectory), map.pgm/map.yaml (Nav2 format), map.png.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

RES = 0.05
MAX_RANGE = 8.0


# ---------------------------------------------------------------- SE(2)
def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def relative(a, b):
    return compose(inverse(a), b)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def tilt_of(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return max(abs(roll), abs(pitch))


# ---------------------------------------------------------------- scans
class ScanConverter:
    """LaserScan -> points (N x 2) in the vehicle frame (odometry point = camera)."""

    def __init__(self, cfg):
        from tt_obstacle_course.scan_utils import ScanView
        self.ScanView = ScanView
        flat = [float(v) for v in cfg.get('scan_self_mask', [])]
        self.mask = [tuple(flat[i:i + 3]) for i in range(0, len(flat) - 2, 3)]
        self.yaw = float(cfg.get('scan_yaw_offset', 0.0))
        self.flip = bool(cfg.get('scan_flip', False))
        self.off = (float(cfg.get('laser_offset_x', -0.15)), float(cfg.get('laser_offset_y', 0.0)))

    def __call__(self, msg):
        v = self.ScanView.from_msg(msg, self.yaw, self.flip, self_mask=self.mask)
        ok = np.isfinite(v.ranges) & (v.ranges > 0.15) & (v.ranges < MAX_RANGE)
        a, r = v.angles[ok], v.ranges[ok]
        return np.stack([r * np.cos(a) + self.off[0], r * np.sin(a) + self.off[1]], 1)


def subsample(pts, n=180):
    if len(pts) <= n:
        return pts
    return pts[np.linspace(0, len(pts) - 1, n).astype(int)]


class StaticFilter:
    """Drops returns that move with the truck. A level lidar hitting a slope (ramp ahead,
    helix coming down into the tunnel) sees a 'wall' at a fixed distance ahead; mapped, it
    smeared across the spot where the tunnel passes under the bridge and held the localizer
    there. Each scan is compared with the latest one taken >= `back` m earlier (moved by
    odometry): a real wall point has an old point next to it, a slope hit doesn't."""

    def __init__(self, back=0.25, max_age=3.0, tol=0.10):
        self.hist = []
        self.back, self.max_age, self.tol = back, max_age, tol

    def __call__(self, t, odom, pts):
        self.hist = [h for h in self.hist if t - h[0] <= self.max_age]
        ref = None
        for h in reversed(self.hist):
            if math.hypot(odom[0] - h[1][0], odom[1] - h[1][1]) >= self.back:
                ref = h
                break
        if not self.hist or math.hypot(odom[0] - self.hist[-1][1][0],
                                       odom[1] - self.hist[-1][1][1]) >= 0.05:
            self.hist.append((t, odom, pts))
        if ref is None or len(ref[2]) < 10 or len(pts) < 10:
            return pts
        d = relative(odom, ref[1])
        c, s = math.cos(d[2]), math.sin(d[2])
        p0 = ref[2]
        old = np.stack([d[0] + c * p0[:, 0] - s * p0[:, 1], d[1] + s * p0[:, 0] + c * p0[:, 1]], 1)
        nn = np.sqrt(((pts[:, None, :] - old[None, :, :]) ** 2).sum(2).min(1))
        far = np.hypot(pts[:, 0], pts[:, 1]) > MAX_RANGE - 0.5
        return pts[(nn < self.tol) | far]


# ---------------------------------------------------------------- likelihood field
class Field:
    """Score lookup: exp(-d^2/2s^2), d = distance to the nearest map wall, at half-cell res."""

    def __init__(self, occ, origin, res, sigma=0.06, inlier=0.10, known=None):
        self.res = res / 2.0
        self.origin = origin
        free = np.where(occ, 0, 255).astype(np.uint8)
        d = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32) * res
        d = cv2.resize(d, (d.shape[1] * 2, d.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        self.score = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        self.inlier_mask = d < inlier
        self.h, self.w = self.score.shape
        self.known = None
        if known is not None:
            k = cv2.resize(known.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            self.known = k.astype(bool) | self.inlier_mask

    def cells(self, x, y):
        ix = np.floor((x - self.origin[0]) / self.res).astype(np.int32)
        iy = np.floor((y - self.origin[1]) / self.res).astype(np.int32)
        ok = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        return np.clip(ix, 0, self.w - 1), np.clip(iy, 0, self.h - 1), ok

    def quality(self, pts, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        ix, iy, ok = self.cells(pose[0] + c * pts[:, 0] - s * pts[:, 1],
                                pose[1] + s * pts[:, 0] + c * pts[:, 1])
        return float(np.mean(self.inlier_mask[iy, ix] & ok)) if len(pts) else 0.0

    def overlap_quality(self, pts, pose):
        """(inlier fraction among points landing on already-mapped area, that area's share).
        Points in unmapped space are new territory, not evidence of a bad pose."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        ix, iy, ok = self.cells(pose[0] + c * pts[:, 0] - s * pts[:, 1],
                                pose[1] + s * pts[:, 0] + c * pts[:, 1])
        on = ok & self.known[iy, ix]
        n = int(on.sum())
        if not n:
            return 1.0, 0.0
        return float(np.mean(self.inlier_mask[iy, ix][on])), n / len(pts)

    def match(self, pts, guess, win_xy=0.2, win_yaw=math.radians(4), step_xy=0.025,
              step_yaw=math.radians(0.5), tau=0.015, sigma=None):
        """Correlative search around guess, soft argmax. sigma = (m, rad) odometry prior:
        where the scan can't tell (along a straight lane, round a curved one) the estimate
        stays at the odometry guess instead of sliding."""
        offs = np.arange(-win_xy, win_xy + 1e-9, step_xy)
        dx, dy = [g.ravel() for g in np.meshgrid(offs, offs)]
        yaws = np.arange(-win_yaw, win_yaw + 1e-9, step_yaw)
        scores = np.empty((len(yaws), len(dx)), np.float32)
        for i, dyaw in enumerate(yaws):
            c, s = math.cos(guess[2] + dyaw), math.sin(guess[2] + dyaw)
            px = guess[0] + c * pts[:, 0] - s * pts[:, 1]
            py = guess[1] + s * pts[:, 0] + c * pts[:, 1]
            ix, iy, ok = self.cells(px[None, :] + dx[:, None], py[None, :] + dy[:, None])
            scores[i] = (self.score[iy, ix] * ok).mean(1)
        best = float(scores.max())
        logw = (scores - best) / tau
        if sigma is not None:
            logw = logw - 0.5 * ((dx ** 2 + dy ** 2)[None, :] / sigma[0] ** 2
                                 + (yaws ** 2)[:, None] / sigma[1] ** 2)
        w = np.exp(logw - logw.max())
        w *= w >= 0.05
        w /= w.sum()
        wy = w.sum(1)
        wxy = w.sum(0)
        pose = (guess[0] + float(wxy @ dx), guess[1] + float(wxy @ dy),
                wrap(guess[2] + float(wy @ yaws)))
        return pose, best


def prior(delta):
    """Odometry uncertainty for a step. Heading is tight: gyro yaw is good step to step, and
    a curved lane lets the scan rotate freely (the narrow curve slid the map 25 deg)."""
    return (0.05 + 0.2 * math.hypot(delta[0], delta[1]),
            math.radians(0.3) + 0.05 * abs(delta[2]))


def robust_match(field, pts, guess, step):
    """Prior-weighted match; if the scan fits badly there (odometry jumped), search wide."""
    pose, _ = field.match(pts, guess, sigma=prior(step))
    q = field.quality(pts, pose)
    if q < 0.6:
        wide, _ = field.match(pts, guess, win_xy=0.4, win_yaw=math.radians(20),
                              step_xy=0.05, step_yaw=math.radians(1))
        wide, _ = field.match(pts, wide)
        wq = field.quality(pts, wide)
        if wq > q + 0.2:
            return wide, wq
    return pose, q


def grid_offset(pts, yaw, tol=math.radians(35), k=8):
    """Heading error vs the course's 90 deg grid, from straight wall runs in the scan
    (points in vehicle frame, ordered by bearing). None unless walls clearly agree."""
    if len(pts) < 40:
        return None
    win = np.lib.stride_tricks.sliding_window_view(pts, (k + 1, 2))[:, 0]   # (M, k+1, 2)
    a, b = win[:, 0], win[:, -1]
    d = b - a
    n = np.hypot(d[:, 0], d[:, 1])
    ok = (n > 0.2) & (n < 1.0)
    if ok.sum() < 10:
        return None
    win, a, d, n = win[ok], a[ok], d[ok], n[ok]
    # Every point in the window within 2 cm of the chord: a straight wall piece.
    rel = win - a[:, None, :]
    dev = np.abs(rel[:, :, 0] * d[:, None, 1] - rel[:, :, 1] * d[:, None, 0]) / n[:, None]
    straight = dev.max(1) < 0.02
    if straight.sum() < 8:
        return None
    ang = np.arctan2(d[straight, 1], d[straight, 0]) + yaw
    q = np.remainder(ang + math.pi / 4, math.pi / 2) - math.pi / 4
    near = np.abs(q) < tol
    if near.sum() < 8 or near.mean() < 0.6:
        return None
    off = math.atan2(np.mean(np.sin(4 * q[near])), np.mean(np.cos(4 * q[near]))) / 4
    return off if np.std(q[near] - off) < math.radians(3) else None


# ---------------------------------------------------------------- map building
class Grid:
    def __init__(self, center, size_m=60.0, res=RES):
        self.res = res
        self.n = int(size_m / res)
        self.origin = (center[0] - size_m / 2, center[1] - size_m / 2)
        self.hits = np.zeros((self.n, self.n), np.float32)
        self.free = np.zeros((self.n, self.n), np.float32)

    def cell(self, x, y):
        return (np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int32),
                np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int32))

    def add(self, pose, pts, sensor):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        wx = pose[0] + c * pts[:, 0] - s * pts[:, 1]
        wy = pose[1] + s * pts[:, 0] + c * pts[:, 1]
        ix, iy = self.cell(wx, wy)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        ix, iy = ix[ok], iy[ok]
        if len(ix) < 3:
            return
        sx, sy = self.cell(pose[0] + c * sensor[0] - s * sensor[1],
                           pose[1] + s * sensor[0] + c * sensor[1])
        x0, x1 = max(0, min(ix.min(), sx) - 1), min(self.n, max(ix.max(), sx) + 2)
        y0, y1 = max(0, min(iy.min(), sy) - 1), min(self.n, max(iy.max(), sy) + 2)
        tmp = np.zeros((y1 - y0, x1 - x0), np.uint8)
        # Free space: triangles sensor -> consecutive returns, not across gaps in the scan.
        p = np.stack([ix - x0, iy - y0], 1)
        jump = np.hypot(*(p[1:] - p[:-1]).T) * self.res
        tris = [np.array([[sx - x0, sy - y0], p[k], p[k + 1]], np.int32)
                for k in np.nonzero(jump < 0.3)[0]]
        if tris:
            cv2.fillPoly(tmp, tris, 1)
        tmp[iy - y0, ix - x0] = 0
        self.free[y0:y1, x0:x1] += tmp
        np.add.at(self.hits, (iy, ix), 1.0)

    def occupied(self):
        return (self.hits >= 2) & (self.hits >= 0.35 * (self.hits + self.free))

    def known(self):
        return (self.hits + self.free) > 0

    def bbox(self, margin=1.0):
        ys, xs = np.nonzero(self.known())
        if not len(xs):
            return 0, self.n, 0, self.n
        m = int(margin / self.res)
        return (max(0, xs.min() - m), min(self.n, xs.max() + m + 1),
                max(0, ys.min() - m), min(self.n, ys.max() + m + 1))


class Mapper:
    """Odometry-seeded scan-to-map matching (SLAM front end) + occupancy grid."""

    def __init__(self, sensor_xy, min_move=0.05, min_turn=math.radians(2), grid_gain=0.15,
                 match=True, q_add=0.6):
        self.grid = None
        self.sensor = sensor_xy
        self.grid_gain = grid_gain
        self.do_match = match
        self.q_add = q_add if match else 0.0
        self.pose = None
        self.last_odom = None
        self.last_added = None
        self.field = None
        self.added = 0
        self.min_move, self.min_turn = min_move, min_turn
        self.traj = []      # (t, x, y, yaw, quality, label)
        self.scans = []     # points of each traj entry (None while the lidar was gated)

    def update(self, t, odom, pts, label=''):
        """pts empty = lidar not usable (tilted): odometry carries the pose, map untouched."""
        if self.pose is None:
            self.grid = Grid(odom[:2])
            self.pose = odom
        else:
            self.pose = compose(self.pose, relative(self.last_odom, odom))
        self.last_odom = odom
        step = (0.0, 0.0, 0.0)
        if self.last_added is not None:
            step = relative(self.last_added, self.pose)
            if math.hypot(step[0], step[1]) < self.min_move and abs(step[2]) < self.min_turn:
                return False
        q = float('nan')
        if len(pts) >= 30:
            q = 1.0
            if self.field is not None and self.do_match:
                sub = subsample(pts)
                pose, q = robust_match(self.field, sub, self.pose, step)
                if q >= 0.3:
                    self.pose = pose
            off = grid_offset(pts, self.pose[2]) if self.grid_gain and self.do_match else None
            if off is not None:
                self.pose = (self.pose[0], self.pose[1], wrap(self.pose[2] - self.grid_gain * off))
            # A scan that disagrees with what is already mapped (odometry jump, truck rocking
            # against a wall) would draw a second copy of the walls: keep the pose, skip it.
            ok = True
            if self.field is not None and self.do_match:
                oq, ov = self.field.overlap_quality(subsample(pts), self.pose)
                ok = ov < 0.3 or oq >= self.q_add
            if ok:
                self.grid.add(self.pose, pts, self.sensor)
                self.added += 1
                if self.added % 10 == 1:
                    self.rebuild_field()
            else:
                pts = np.zeros((0, 2))
        self.last_added = self.pose
        self.traj.append((t, *self.pose, q, label))
        self.scans.append(np.asarray(pts, np.float32) if len(pts) >= 30 else None)
        return True

    def path_length(self):
        p = np.array([f[1:3] for f in self.traj])
        return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0

    def close_loop(self, pts, min_path=20.0, start_frac=0.25, q_ok=0.7):
        """Back where mapping began (the start/finish line): match the scan against the start
        area alone, then spread the end error back over the lap so start and finish meet.
        Returns (moved_m, turned_deg, quality) or None if it didn't match well enough."""
        if len(self.traj) < 20 or pts is None or len(pts) < 30:
            return None
        p = np.array([f[1:4] for f in self.traj])
        seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < min_path:
            return None
        cur = self.pose
        start = [i for i in range(len(p)) if s[i] <= start_frac * total and self.scans[i] is not None
                 and math.hypot(p[i, 0] - cur[0], p[i, 1] - cur[1]) < 4.0]
        if len(start) < 3:
            return None
        g = Grid(cur[:2], 20.0)
        for i in start:
            g.add(tuple(p[i]), self.scans[i], self.sensor)
        x0, x1, y0, y1 = g.bbox(margin=2.0)
        field = Field(g.occupied()[y0:y1, x0:x1], (g.origin[0] + x0 * RES, g.origin[1] + y0 * RES), RES)
        sub = subsample(pts)
        coarse, _ = field.match(subsample(pts, 90), cur, win_xy=1.5, win_yaw=math.radians(20),
                                step_xy=0.05, step_yaw=math.radians(1))
        target, _ = field.match(sub, coarse)
        q = field.quality(sub, target)
        if q < q_ok:
            return None
        # Spread the heading error over the lap by path length, then the leftover offset.
        dyaw = wrap(target[2] - cur[2])
        steps = [relative(tuple(p[i - 1]), tuple(p[i])) for i in range(1, len(p))]
        new = [tuple(p[0])]
        for i, r in enumerate(steps):
            new.append(compose(new[-1], (r[0], r[1], r[2] + dyaw * seg[i] / total)))
        new = np.array(new)
        res = np.array(target[:2]) - new[-1, :2]
        new[:, :2] += res[None, :] * (s / total)[:, None]
        self.traj = [(f[0], *new[i], f[4], f[5]) for i, f in enumerate(self.traj)]
        center = new[:, :2].mean(0)
        self.grid = Grid(center, max(60.0, float(np.ptp(new[:, :2], 0).max()) + 20.0))
        for i, sc in enumerate(self.scans):
            if sc is not None:
                self.grid.add(tuple(new[i]), sc, self.sensor)
        moved = math.hypot(target[0] - cur[0], target[1] - cur[1])
        self.pose = self.last_added = tuple(new[-1])
        self.rebuild_field()
        return moved, math.degrees(dyaw), q

    def rebuild_field(self):
        x0, x1, y0, y1 = self.grid.bbox(margin=2.0)
        occ = self.grid.occupied()[y0:y1, x0:x1]
        origin = (self.grid.origin[0] + x0 * RES, self.grid.origin[1] + y0 * RES)
        self.field = Field(occ, origin, RES, known=self.grid.known()[y0:y1, x0:x1])


# ---------------------------------------------------------------- saved map
class CourseMap:
    def __init__(self, occ, known, origin, res, traj=None, labels=None):
        self.occ, self.known, self.origin, self.res = occ, known, tuple(origin), res
        self.traj = np.zeros((0, 3)) if traj is None else np.asarray(traj, float)
        self.labels = [] if labels is None else list(labels)
        self.field = Field(occ, self.origin, res)

    @classmethod
    def from_grid(cls, grid, traj):
        x0, x1, y0, y1 = grid.bbox()
        origin = (grid.origin[0] + x0 * grid.res, grid.origin[1] + y0 * grid.res)
        return cls(grid.occupied()[y0:y1, x0:x1], grid.known()[y0:y1, x0:x1], origin, grid.res,
                   [p[1:4] for p in traj], [p[5] for p in traj])

    @classmethod
    def load(cls, path):
        f = os.path.join(path, 'map.npz') if os.path.isdir(path) else path
        d = np.load(f, allow_pickle=False)
        return cls(d['occ'].astype(bool), d['known'].astype(bool), d['origin'], float(d['res']),
                   d['traj'], [str(s) for s in d['labels']])

    def zone_at(self, x, y, max_d=1.5):
        if not len(self.traj):
            return ''
        d = np.hypot(self.traj[:, 0] - x, self.traj[:, 1] - y)
        i = int(np.argmin(d))
        return self.labels[i] if d[i] <= max_d else ''

    def save(self, out):
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, 'map.npz'), occ=self.occ, known=self.known,
                            origin=np.array(self.origin), res=self.res, traj=self.traj,
                            labels=np.array(self.labels, dtype=str))
        img = np.full(self.occ.shape, 205, np.uint8)
        img[self.known] = 254
        img[self.occ] = 0
        cv2.imwrite(os.path.join(out, 'map.pgm'), img[::-1])
        with open(os.path.join(out, 'map.yaml'), 'w') as f:
            f.write(f'image: map.pgm\nmode: trinary\nresolution: {self.res}\n'
                    f'origin: [{self.origin[0]:.3f}, {self.origin[1]:.3f}, 0.0]\n'
                    'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
        cv2.imwrite(os.path.join(out, 'map.png'), self.preview())

    def preview(self, scale=3):
        img = np.full(self.occ.shape + (3,), 110, np.uint8)
        img[self.known] = (235, 235, 235)
        img[self.occ] = (0, 0, 0)
        # Row 0 is the lowest y: flip so north is up, like the labels and map.pgm.
        img = img[::-1].copy()
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        h = img.shape[0]

        def px(x, y):
            return (int((x - self.origin[0]) / self.res * scale),
                    int(h - (y - self.origin[1]) / self.res * scale))
        prev_label = None
        for (x, y, _), lab in zip(self.traj, self.labels):
            hue = (sum(map(ord, lab)) * 37) % 180
            col = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])
            cv2.circle(img, px(x, y), 2, col, -1)
            if lab != prev_label and lab:
                cv2.putText(img, lab, px(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                prev_label = lab
        for m in range(0, int(self.occ.shape[1] * self.res) + 1, 2):
            cv2.putText(img, f'{m + self.origin[0]:.0f}', (int(m / self.res * scale), h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        return img

    def to_msg(self, stamp=None):
        from nav_msgs.msg import OccupancyGrid
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        if stamp is not None:
            msg.header.stamp = stamp
        msg.info.resolution = float(self.res)
        msg.info.height, msg.info.width = self.occ.shape
        msg.info.origin.position.x, msg.info.origin.position.y = map(float, self.origin)
        msg.info.origin.orientation.w = 1.0
        data = np.full(self.occ.shape, -1, np.int8)
        data[self.known] = 0
        data[self.occ] = 100
        msg.data = data.ravel().tolist()
        return msg


# ---------------------------------------------------------------- localizer
class Localizer:
    """Map pose from odometry deltas, corrected by scan-to-map matching when it fits well."""

    def __init__(self, cmap, q_min=0.6, q_lock=0.75):
        self.map = cmap
        self.q_min = q_min
        self.q_lock = q_lock
        self.pose = None
        self.last_odom = None
        self.quality = 0.0
        self.state = 'init'
        self.last_recover = -1e9

    def global_search(self, pts, around=None):
        """Best pose among the mapping trajectory (or around a given pose) and headings."""
        sub = subsample(pts, 60)
        f = self.map.field
        if around is not None:
            cands = [around]
        else:
            tr = self.map.traj
            keep = [0]
            for i in range(1, len(tr)):
                if math.hypot(*(tr[i, :2] - tr[keep[-1], :2])) >= 0.5:
                    keep.append(i)
            cands = [tuple(tr[i]) for i in keep]
        best = (-1.0, None)
        for c in cands:
            for dyaw in np.radians(np.arange(-20, 21, 10)):
                pose, sc = f.match(sub, (c[0], c[1], c[2] + dyaw), win_xy=0.5, win_yaw=math.radians(3),
                                   step_xy=0.1, step_yaw=math.radians(1.5))
                if sc > best[0]:
                    best = (sc, pose)
        if best[1] is None:
            return None, 0.0
        pose, _ = f.match(subsample(pts), best[1])
        return pose, f.quality(subsample(pts), pose)

    def update(self, odom, pts, now):
        step = (0.0, 0.0, 0.0)
        if self.pose is not None and self.last_odom is not None:
            step = relative(self.last_odom, odom)
            self.pose = compose(self.pose, step)
        self.last_odom = odom
        if pts is None or len(pts) < 30:
            return self.pose
        sub = subsample(pts)
        if self.pose is None:
            # Same start box as the mapping drive: map frame == odometry frame.
            pose = self.map.field.match(sub, odom)[0]
            q = self.map.field.quality(sub, pose)
            if q < self.q_min:
                pose, q = self.global_search(pts)
            if pose is None or q < self.q_min:
                self.state = 'init'
                return None
            self.pose, self.quality, self.state = pose, q, 'tracking'
            return self.pose
        pose, q = robust_match(self.map.field, sub, self.pose, step)
        if q < self.q_min and now - self.last_recover > 1.0:
            # Off by more than the tracking window (e.g. 0.7 m after the helix, lidar blind
            # for 270 deg): re-acquire over a wide area, accept only a clearly good fit.
            self.last_recover = now
            coarse, _ = self.map.field.match(subsample(pts, 90), self.pose, win_xy=1.0,
                                             win_yaw=math.radians(15), step_xy=0.05,
                                             step_yaw=math.radians(1.5))
            fine, _ = self.map.field.match(sub, coarse)
            fq = self.map.field.quality(sub, fine)
            if fq >= self.q_lock:
                pose, q = fine, fq
        self.quality = q
        if q >= self.q_min:
            self.pose = pose
            self.state = 'tracking'
        else:
            self.state = 'odometry only'
        return self.pose


# ---------------------------------------------------------------- bag input
def read_bag(path, topics):
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    sid = ''
    meta = os.path.join(path, 'metadata.yaml')
    if os.path.exists(meta):
        with open(meta) as f:
            sid = yaml.safe_load(f)['rosbag2_bagfile_information'].get('storage_identifier', '')
    elif os.path.isdir(path):
        # Recording cut off (power, crash): no metadata.yaml. Open the data file itself.
        files = sorted(f for f in os.listdir(path) if f.endswith(('.mcap', '.db3')))
        if not files:
            sys.exit(f'no .mcap/.db3 in {path}')
        if len(files) > 1:
            print(f'(unindexed bag: reading only {files[0]}; run `ros2 bag reindex {path}` '
                  'for all of it)', file=sys.stderr)
        sid = 'mcap' if files[0].endswith('.mcap') else 'sqlite3'
        path = os.path.join(path, files[0])
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=sid),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    classes = {t: get_message(types[t]) for t in topics if t in types}
    missing = [t for t in topics if t not in types]
    while reader.has_next():
        topic, data, t = reader.read_next()
        yield topic, deserialize_message(data, classes[topic]), t * 1e-9
    if missing:
        print(f'(not in bag: {", ".join(missing)})', file=sys.stderr)


class Labeler:
    """Zone at each moment of a drive: the mission's zone if it is running, else the last tag."""

    def __init__(self):
        self.zone = ''

    def feed(self, topic, msg):
        if topic == '/mission/state':
            try:
                st = json.loads(msg.data)
            except ValueError:
                return
            if st.get('started') and st.get('zone'):
                self.zone = st['zone']
        elif topic == '/zone_event':
            self.zone = msg.data


def odom_pose(msg):
    """-> (x, y, yaw), tilt, pitch (nose down +)."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return (p.x, p.y, yaw_of(q)), tilt_of(q), pitch


TILT_MAX = math.radians(3.0)


class GyroOdom:
    """Heading from the IMU gyro, distance from odometry. In the 3-D sim VO heading jumped by
    up to 45 deg while the integrated gyro drifted 6 deg in 7 min (scan matching removes that).
    Without IMU messages it passes odometry through unchanged."""

    def __init__(self):
        self.pose = None
        self.prev = None
        self.dyaw = 0.0
        self.imu_t = None
        self.has_imu = False

    def on_imu(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.imu_t is not None and 0.0 < t - self.imu_t < 0.5:
            self.dyaw += msg.angular_velocity.z * (t - self.imu_t)
            self.has_imu = True
        self.imu_t = t

    def __call__(self, odom):
        if not self.has_imu:
            return odom
        if self.pose is None:
            self.pose, self.prev, self.dyaw = odom, odom, 0.0
            return self.pose
        dx, dy = odom[0] - self.prev[0], odom[1] - self.prev[1]
        fwd = dx * math.cos(odom[2]) + dy * math.sin(odom[2])
        ds = math.copysign(math.hypot(dx, dy), fwd)
        mid = self.pose[2] + self.dyaw / 2
        self.pose = (self.pose[0] + ds * math.cos(mid), self.pose[1] + ds * math.sin(mid),
                     wrap(self.pose[2] + self.dyaw))
        self.prev, self.dyaw = odom, 0.0
        return self.pose
# The tunnel runs under the bridge: a 2-D map can hold only the ground level.
LEVEL_DZ = 0.25


class LevelGate:
    """Is the lidar seeing the ground-level course right now (not tilted, not up on the bridge)?
    Height from pitch x distance driven: VO height drifted to -1.3 m on flat ground (3-D sim),
    this stayed within 8 cm over the whole course."""

    def __init__(self):
        self.h = 0.0
        self.prev = None
        self.lidar_ok = True

    def feed(self, pose, pitch):
        if self.prev is not None:
            ds = ((pose[0] - self.prev[0]) * math.cos(pose[2])
                  + (pose[1] - self.prev[1]) * math.sin(pose[2]))
            self.h -= math.sin(pitch) * ds
        self.prev = pose

    def ok(self, tilt):
        return self.lidar_ok and tilt <= TILT_MAX and abs(self.h) <= LEVEL_DZ


class MapSession:
    """Live mapping fed with ROS messages (tt_dash Map button, tt_map live).

    Auto-stop: once the drive is long enough (min_path) and the truck is back within
    `near` m of where mapping started, it tries to close the loop on the start area; on a
    good match the map is finalised and saved, and `done` is set. Saved every 30 s anyway."""

    def __init__(self, cfg, out_dir, auto_stop=True, min_path=20.0, near=2.5, log=print):
        self.conv = ScanConverter(cfg)
        self.mapper = Mapper(self.conv.off)
        self.gate = LevelGate()
        self.gyro = GyroOdom()
        self.filt = StaticFilter()
        self.lab = Labeler()
        self.out = out_dir
        self.auto_stop, self.min_path, self.near = auto_stop, min_path, near
        self.log = log
        self.odom = None
        self.tilt = 0.0
        self.done = False
        self.closed = None
        self.last_try = -1e9
        self.last_save = time.monotonic()
        self.saved = None

    def on_imu(self, msg):
        self.gyro.on_imu(msg)

    def on_odom(self, msg):
        odom, self.tilt, pitch = odom_pose(msg)
        self.odom = self.gyro(odom)
        self.gate.feed(self.odom, pitch)

    def on_scan(self, msg):
        if self.done or self.odom is None:
            return
        now = time.monotonic()
        level = self.gate.ok(self.tilt)
        pts = self.filt(now, self.odom, self.conv(msg)) if level else np.zeros((0, 2))
        self.mapper.update(now, self.odom, pts, self.lab.zone)
        m = self.mapper
        if (self.auto_stop and level and now - self.last_try > 2.0 and len(m.traj) > 20
                and math.hypot(m.pose[0] - m.traj[0][1], m.pose[1] - m.traj[0][2]) < self.near
                and m.path_length() >= self.min_path):
            self.last_try = now
            res = m.close_loop(pts, self.min_path)
            if res is not None:
                self.closed = res
                self.log(f'map: back at the start, loop closed ({res[0]:.2f} m, {res[1]:+.1f} deg '
                         f'end error spread over the lap, quality {res[2]:.2f}); saving')
                self.stop()
                return
        if now - self.last_save > 30.0:
            self.save()

    def course_map(self):
        m = self.mapper
        return CourseMap.from_grid(m.grid, m.traj) if m.grid is not None and m.traj else None

    def save(self):
        cm = self.course_map()
        self.last_save = time.monotonic()
        if cm is None:
            return None
        cm.save(self.out)
        self.saved = self.out
        return self.out

    def stop(self):
        self.done = True
        return self.save()

    def status(self):
        m = self.mapper
        q = [f[4] for f in m.traj[-20:] if f[4] == f[4]]
        return {'mapping': not self.done, 'dir': self.out, 'scans': m.added,
                'path_m': round(m.path_length(), 1), 'quality': round(float(np.mean(q)), 2) if q else None,
                'loop': None if self.closed is None else [round(v, 2) for v in self.closed],
                'saved': self.saved}


def build(args, cfg):
    conv = ScanConverter(cfg)
    mapper = Mapper(conv.off, match=not args.trust_odom)
    lab = Labeler()
    gate = LevelGate()
    filt = StaticFilter()
    odom, tilt = None, 0.0
    n_scans = skipped = 0
    t_start = time.time()
    gyro = GyroOdom()
    for topic, msg, t in read_bag(args.bag, [args.scan, args.odom, '/terrain/lidar_ok',
                                             '/mission/state', '/zone_event', '/zed/zed_node/imu/data']):
        if topic == '/zed/zed_node/imu/data':
            if not args.trust_odom:
                gyro.on_imu(msg)
        elif topic == args.odom:
            odom, tilt, pitch = odom_pose(msg)
            odom = gyro(odom)
            gate.feed(odom, pitch)
        elif topic == '/terrain/lidar_ok':
            gate.lidar_ok = msg.data
        elif topic == args.scan:
            n_scans += 1
            if odom is None:
                continue
            if not gate.ok(tilt):
                skipped += 1
                mapper.update(t, odom, np.zeros((0, 2)), lab.zone)
                continue
            mapper.update(t, odom, filt(t, odom, conv(msg)), lab.zone)
        else:
            lab.feed(topic, msg)
    if mapper.grid is None:
        sys.exit('no scans with odometry in the bag')
    cmap = CourseMap.from_grid(mapper.grid, mapper.traj)
    out = os.path.expanduser(os.path.join(args.dir, args.out))
    cmap.save(out)
    q = np.array([p[4] for p in mapper.traj])
    print(f'{n_scans} scans, {mapper.added} added, {skipped} skipped (tilted) in {time.time() - t_start:.0f} s\n'
          f'map {cmap.occ.shape[1] * cmap.res:.1f} x {cmap.occ.shape[0] * cmap.res:.1f} m, '
          f'match quality median {np.nanmedian(q):.2f}, <0.5 in {np.mean(q[np.isfinite(q)] < 0.5) * 100:.0f}% of scans\n'
          f'zones: {" ".join(dict.fromkeys(l for l in cmap.labels if l))}\n'
          f'saved {out}/  (map.png to check it)')


def evaluate(args, cfg):
    cmap = CourseMap.load(os.path.expanduser(args.map))
    conv = ScanConverter(cfg)
    loc = Localizer(cmap)
    gate = LevelGate()
    filt = StaticFilter()
    odom = truth = None
    tilt = 0.0
    rows = []
    gyro = GyroOdom()
    for topic, msg, t in read_bag(args.bag, [args.scan, args.odom, args.truth, '/terrain/lidar_ok', '/zed/zed_node/imu/data']):
        if topic == '/zed/zed_node/imu/data':
            gyro.on_imu(msg)
        elif topic == args.odom:
            raw, tilt, pitch = odom_pose(msg)
            odom = gyro(raw)
            gate.feed(odom, pitch)
        elif topic == args.truth:
            truth = odom_pose(msg)[0]
        elif topic == '/terrain/lidar_ok':
            gate.lidar_ok = msg.data
        elif topic == args.scan and odom is not None:
            level = gate.ok(tilt)
            pose = loc.update(odom, filt(t, odom, conv(msg)) if level else None, t)
            if pose is not None and truth is not None:
                rows.append((t, math.hypot(pose[0] - truth[0], pose[1] - truth[1]),
                             math.hypot(raw[0] - truth[0], raw[1] - truth[1]),
                             math.degrees(abs(wrap(pose[2] - truth[2]))), loc.quality,
                             cmap.zone_at(pose[0], pose[1]), pose[:2], truth[:2]))
            elif pose is not None:
                rows.append((t, float('nan'), float('nan'), float('nan'), loc.quality,
                             cmap.zone_at(pose[0], pose[1]), None, None))
    if not rows:
        sys.exit('localizer never initialised')
    a = np.array([r[1:5] for r in rows], float)
    print(f'{len(rows)} scans localized; quality median {np.nanmedian(a[:, 3]):.2f}')
    if np.isfinite(a[:, 0]).any():
        for name, col in (('map pose', 0), ('odometry', 1)):
            e = a[:, col]
            print(f'{name:9} error: median {np.nanmedian(e):.2f} m, 95% {np.nanpercentile(e, 95):.2f} m, '
                  f'max {np.nanmax(e):.2f} m')
        print(f'map heading error: median {np.nanmedian(a[:, 2]):.1f} deg, max {np.nanmax(a[:, 2]):.1f} deg')
        # The map frame itself may sit at a fixed offset from the truth frame (mapping drive's
        # own start error); what matters is consistency, so also report after a rigid fit.
        P = np.array([r[6] for r in rows if r[6] is not None])
        T = np.array([r[7] for r in rows if r[6] is not None])
        if len(P) > 10:
            pc, tc = P.mean(0), T.mean(0)
            u, _, vt = np.linalg.svd((P - pc).T @ (T - tc))
            R = (u @ vt).T
            if np.linalg.det(R) < 0:
                vt[-1] *= -1
                R = (u @ vt).T
            fit = (R @ (P - pc).T).T + tc
            e = np.hypot(*(fit - T).T)
            print(f'after rigid fit ({math.degrees(math.atan2(R[1, 0], R[0, 0])):+.1f} deg, '
                  f'{np.hypot(*(tc - R @ pc)):.2f} m): median {np.median(e):.2f} m, 95% '
                  f'{np.percentile(e, 95):.2f} m, max {e.max():.2f} m')
        zones = {}
        for r in rows:
            zones.setdefault(r[5], []).append(r[1])
        for z, e in zones.items():
            print(f'  {z or "-":14} {len(e):5} scans  map error median {np.median(e):.2f} max {np.max(e):.2f} m')


def update(args, cfg):
    """Localize a recorded run on the map and redo the chosen zones' cells from that run."""
    base = CourseMap.load(os.path.expanduser(args.map))
    zones = set(args.zones)
    every = 'all' in zones
    known_zones = set(l for l in base.labels if l)
    if not every and zones - known_zones:
        sys.exit(f'zones not in the map: {sorted(zones - known_zones)}; map has {sorted(known_zones)}')
    h, w = base.occ.shape
    margin = 2.0
    size = max(h, w) * base.res + 2 * margin
    grid = Grid((base.origin[0] - margin + size / 2, base.origin[1] - margin + size / 2), size, base.res)
    conv = ScanConverter(cfg)
    loc = Localizer(base)
    gate = LevelGate()
    filt = StaticFilter()
    odom, tilt = None, 0.0
    used = skipped = 0
    gyro = GyroOdom()
    for topic, msg, t in read_bag(args.bag, [args.scan, args.odom, '/terrain/lidar_ok', '/zed/zed_node/imu/data']):
        if topic == '/zed/zed_node/imu/data':
            gyro.on_imu(msg)
        elif topic == args.odom:
            odom, tilt, pitch = odom_pose(msg)
            odom = gyro(odom)
            gate.feed(odom, pitch)
        elif topic == '/terrain/lidar_ok':
            gate.lidar_ok = msg.data
        elif odom is not None:
            level = gate.ok(tilt)
            pts = filt(t, odom, conv(msg)) if level else None
            pose = loc.update(odom, pts, t)
            if pose is None or pts is None or loc.state != 'tracking':
                skipped += 1
                continue
            if every or base.zone_at(pose[0], pose[1]) in zones:
                grid.add(pose, pts, conv.off)
                used += 1
    if not used:
        sys.exit('no well-localized scans in those zones; nothing updated')
    oy = int(round((base.origin[1] - grid.origin[1]) / base.res))
    ox = int(round((base.origin[0] - grid.origin[0]) / base.res))
    seen = (grid.hits + grid.free)[oy:oy + h, ox:ox + w] >= 3
    new_occ = grid.occupied()[oy:oy + h, ox:ox + w]
    occ, known = base.occ.copy(), base.known.copy()
    added = int((seen & new_occ & ~occ).sum())
    removed = int((seen & ~new_occ & occ).sum())
    occ[seen] = new_occ[seen]
    known |= seen
    out_map = CourseMap(occ, known, base.origin, base.res, base.traj, base.labels)
    out = os.path.expanduser(os.path.join(args.dir, args.out))
    out_map.save(out)
    print(f'{used} scans used, {skipped} not localized/level; {int(seen.sum())} cells re-observed: '
          f'{added} newly occupied, {removed} cleared ({added * base.res ** 2:.2f} / '
          f'{removed * base.res ** 2:.2f} m2)\nsaved {out}/  (original {args.map} untouched)')


# ---------------------------------------------------------------- ROS nodes
def ros_common():
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    return rclpy, latched


def pose_msg(pose, stamp, frame='map'):
    from geometry_msgs.msg import PoseStamped
    m = PoseStamped()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.pose.position.x, m.pose.position.y = float(pose[0]), float(pose[1])
    m.pose.orientation.z, m.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
    return m


def start_bag(out):
    """Record the same topics as tt_record into <out>/bag (for rebuilding the map later)."""
    import shutil
    import subprocess
    from tt_obstacle_course.record import TOPICS
    if shutil.which('ros2') is None:
        return None
    log = open(os.path.join(out, 'bag.log'), 'w')
    return subprocess.Popen(['ros2', 'bag', 'record', '-o', os.path.join(out, 'bag')] + TOPICS,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_bag(rec, wait=8.0):
    import signal
    import subprocess
    if rec is None or rec.poll() is not None:
        return
    os.killpg(rec.pid, signal.SIGINT)
    try:
        rec.wait(wait)
    except subprocess.TimeoutExpired:
        os.killpg(rec.pid, signal.SIGKILL)


def live(args, cfg):
    """Map while driving; /map republished every 2 s; stops itself back at the start line
    (unless --no-auto-stop). Ctrl-C (or the launch shutting down) saves the map."""
    rclpy, latched = ros_common()
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Bool, String
    from tt_obstacle_course.debug_dash import workspace_maps_dir
    rclpy.init(args=['tt_map'] + getattr(args, 'ros_args', []))
    node = rclpy.create_node('tt_map')
    out = os.path.join(os.path.expanduser(args.dir) if args.dir else workspace_maps_dir(), args.out)
    os.makedirs(out, exist_ok=True)
    ses = MapSession(cfg, out, auto_stop=not args.no_auto_stop,
                     log=lambda s: node.get_logger().info(s))
    rec = start_bag(out) if args.record else None
    pub_map = node.create_publisher(OccupancyGrid, '/map', latched)
    pub_pose = node.create_publisher(PoseStamped, '/localization/pose', 10)

    def on_scan(m):
        ses.on_scan(m)
        if ses.mapper.pose is not None:
            pub_pose.publish(pose_msg(ses.mapper.pose, m.header.stamp))
        if ses.done:
            raise KeyboardInterrupt

    def publish():
        cm = ses.course_map()
        if cm is not None:
            pub_map.publish(cm.to_msg(node.get_clock().now().to_msg()))

    node.create_subscription(Odometry, args.odom, ses.on_odom, 20)
    node.create_subscription(Imu, '/zed/zed_node/imu/data', ses.on_imu, qos_profile_sensor_data)
    node.create_subscription(LaserScan, args.scan, on_scan, qos_profile_sensor_data)
    node.create_subscription(Bool, '/terrain/lidar_ok', lambda m: setattr(ses.gate, 'lidar_ok', m.data), 10)
    node.create_subscription(String, '/mission/state', lambda m: ses.lab.feed('/mission/state', m), 10)
    node.create_subscription(String, '/zone_event', lambda m: ses.lab.feed('/zone_event', m), 10)
    node.create_timer(2.0, publish)
    node.get_logger().info(f'mapping; drive the course slowly. '
                           f'{"Ctrl-C saves" if args.no_auto_stop else "Stops back at the start, or Ctrl-C"}'
                           f'{" (recording bag)" if rec else ""}. -> {out}')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        path = ses.stop()
        print(f'[tt_map] map saved: {path} ({ses.mapper.added} scans)', flush=True)
        stop_bag(rec)
        if rec is not None:
            print(f'[tt_map] bag saved: {out}/bag', flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main_localizer(args=None):
    """map_localizer node: params map_file, q_min, plus the tuning scan/laser settings."""
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Bool, String

    from tt_obstacle_course.ros_common import TTNode, dumps, run

    class MapLocalizer(TTNode):
        def __init__(self):
            super().__init__('map_localizer')
            _, latched = ros_common()
            path = os.path.expanduser(str(self.param('map_file', '')))
            self.map = CourseMap.load(path)
            self.loc = Localizer(self.map, float(self.param('q_min', 0.6)))
            g = self.param
            self.conv = ScanConverter({
                'scan_self_mask': g('scan_self_mask', [0.0]), 'scan_yaw_offset': g('scan_yaw_offset', 0.0),
                'scan_flip': g('scan_flip', False), 'laser_offset_x': g('laser_offset_x', -0.15),
                'laser_offset_y': g('laser_offset_y', 0.0)})
            self.local = None
            self.tilt = 0.0
            self.gate = LevelGate()
            self.gyro = GyroOdom()
            self.filt = StaticFilter()
            self.last_state = None
            self.pub_pose = self.create_publisher(PoseStamped, '/localization/pose', 10)
            self.pub_status = self.create_publisher(String, '/localization/status', 10)
            self.pub_corr = self.create_publisher(String, '/localization/correction', 10)
            self.pub_map = self.create_publisher(OccupancyGrid, '/map', latched)
            self.pub_map.publish(self.map.to_msg(self.get_clock().now().to_msg()))
            # Odometry the map correction is NOT applied to (odom_guard), so no feedback loop.
            self.create_subscription(Odometry, self.param('local_odom_topic', '/odom/local'),
                                     self.on_odom, 20)
            self.create_subscription(Imu, self.param('imu_topic', '/zed/zed_node/imu/data'),
                                     self.gyro.on_imu, qos_profile_sensor_data)
            self.create_subscription(LaserScan, self.param('scan_topic', '/scan'), self.on_scan,
                                     qos_profile_sensor_data)
            self.create_subscription(Bool, '/terrain/lidar_ok',
                                     lambda m: setattr(self.gate, 'lidar_ok', m.data), 10)
            self.get_logger().info(f'map {path}: {self.map.occ.shape[1] * self.map.res:.1f} x '
                                   f'{self.map.occ.shape[0] * self.map.res:.1f} m')

        def on_odom(self, m):
            self.raw, self.tilt, pitch = odom_pose(m)
            self.local = self.gyro(self.raw)
            self.gate.feed(self.local, pitch)

        def on_scan(self, m):
            if self.local is None:
                return
            level = self.gate.ok(self.tilt)
            t0 = time.monotonic()
            now = self.now_s()
            pts = self.filt(now, self.local, self.conv(m)) if level else None
            pose = self.loc.update(self.local, pts, now)
            if self.loc.state != self.last_state:
                self.get_logger().info(f'map localization: {self.loc.state} (quality {self.loc.quality:.2f})')
                self.last_state = self.loc.state
            if pose is None:
                return
            zone = self.map.zone_at(pose[0], pose[1])
            self.pub_pose.publish(pose_msg(pose, m.header.stamp))
            self.pub_status.publish(String(data=dumps({
                'state': self.loc.state, 'quality': round(self.loc.quality, 2), 'zone': zone,
                'pose': [round(v, 3) for v in pose], 'ms': round((time.monotonic() - t0) * 1000, 1)})))
            if level and self.loc.state == 'tracking':
                self.pub_corr.publish(String(data=dumps({
                    'local': list(self.raw), 'map': list(pose), 'q': self.loc.quality})))

    run(MapLocalizer, args)


def main(argv=None):
    raw = sys.argv[1:] if argv is None else list(argv)
    ros_args = []
    if '--ros-args' in raw:
        i = raw.index('--ros-args')
        raw, ros_args = raw[:i], raw[i:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('build', 'live', 'show', 'eval', 'update'):
        p = sub.add_parser(name)
        p.add_argument('--tuning')
        p.add_argument('--scan', default='/scan')
        p.add_argument('--odom', default='/odom/local',
                       help='odometry without map correction (falls back to /odom/fused in bags without it)')
        if name in ('build', 'live', 'update'):
            p.add_argument('-o', '--out', default=time.strftime('map_%m%d_%H%M%S'))
            p.add_argument('--dir', default=None if name == 'live' else '~/maps',
                           help='(live: default <workspace>/maps, same as the dashboard)')
        if name == 'live':
            p.add_argument('--record', action='store_true', help='also record a bag into <map>/bag')
            p.add_argument('--no-auto-stop', action='store_true',
                           help='keep mapping past the start line (several laps); Ctrl-C saves')
        if name == 'build':
            p.add_argument('bag')
            p.add_argument('--trust-odom', action='store_true',
                           help='no scan matching: odometry poses as they are (ground-truth tests)')
        if name in ('show', 'eval', 'update'):
            p.add_argument('map')
        if name == 'eval':
            p.add_argument('bag')
            p.add_argument('--truth', default='/sim/odom_truth')
        if name == 'update':
            p.add_argument('bag')
            p.add_argument('--zones', nargs='+', required=True,
                           help="zones to re-map from this run (names from the map), or 'all'")
    a = ap.parse_args(raw)
    a.ros_args = ros_args
    from tt_obstacle_course.debug_dash import load_tuning
    cfg = load_tuning(a.tuning)
    if a.cmd in ('build', 'eval', 'update'):
        a.odom = pick_odom(a.bag, a.odom)
    if a.cmd == 'build':
        build(a, cfg)
    elif a.cmd == 'update':
        update(a, cfg)
    elif a.cmd == 'eval':
        evaluate(a, cfg)
    elif a.cmd == 'live':
        live(a, cfg)
    else:
        cm = CourseMap.load(os.path.expanduser(a.map))
        out = os.path.dirname(os.path.join(os.path.expanduser(a.map), 'map.npz'))
        cv2.imwrite(os.path.join(out, 'map.png'), cm.preview())
        print(f'{cm.occ.shape[1] * cm.res:.1f} x {cm.occ.shape[0] * cm.res:.1f} m at {cm.res} m, '
              f'origin {cm.origin}, {len(cm.traj)} trajectory points, zones: '
              f'{" ".join(dict.fromkeys(l for l in cm.labels if l))}\nredrew {out}/map.png')


def pick_odom(bag, wanted):
    try:
        import yaml
        with open(os.path.join(bag, 'metadata.yaml')) as f:
            info = yaml.safe_load(f)['rosbag2_bagfile_information']
        names = [t['topic_metadata']['name'] for t in info['topics_with_message_count']]
    except (OSError, KeyError, TypeError):
        return wanted
    if wanted in names:
        return wanted
    for alt in ('/odom/local', '/odom/fused'):
        if alt in names:
            print(f'(odometry: {alt})', file=sys.stderr)
            return alt
    return wanted


if __name__ == '__main__':
    main()
