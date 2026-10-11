"""Compact Hybrid-A* for the rear-axle bicycle model, planned on lidar points in the
odom frame. Pure logic, no ROS. Poses are REAR-AXLE poses (x, y, yaw).

The truck is covered by circles along its centre line (rear axle to front bumper);
obstacles are inflated by the circle radius, so each circle centre only needs a
grid lookup. Primitives are full-lock left/right and straight arcs, forward and
reverse, so every path respects the minimum turning radius.
"""

import heapq
import math

import numpy as np


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ObstacleMap:
    """Lidar returns (odom frame) on a coarse cell set. `cells` are permanent (virtual
    fences, stall marks); scan returns time-stamped via add_scan(t=...) expire after
    `ttl` s unless seen again, so returns logged from a drifted pose don't build up
    into ghost walls (buckets smeared shut under VO drift). Nearby obstacles are
    re-seen every scan by the 360 deg lidar."""

    def __init__(self, res=0.05, max_range=5.0, ttl=6.0):
        self.res = res
        self.max_range = max_range
        self.ttl = ttl
        self.cells = set()
        self.seen = {}
        self.t = 0.0

    def add_scan(self, scan, x, y, yaw, t=None):
        """scan: ScanView in the lidar frame; (x, y, yaw): lidar pose in odom."""
        xs, ys = scan.points()
        keep = np.hypot(xs, ys) < self.max_range
        c, s = math.cos(yaw), math.sin(yaw)
        ox = x + c * xs[keep] - s * ys[keep]
        oy = y + s * xs[keep] + c * ys[keep]
        new = zip(np.floor(ox / self.res).astype(int).tolist(),
                  np.floor(oy / self.res).astype(int).tolist())
        if t is None:
            self.cells.update(new)
            return
        self.t = max(self.t, t)
        self.seen.update((cell, t) for cell in new)
        if len(self.seen) > 20000:
            cut = self.t - self.ttl
            self.seen = {k: v for k, v in self.seen.items() if v >= cut}

    def occupied(self):
        cut = self.t - self.ttl
        return self.cells | {k for k, v in self.seen.items() if v >= cut}

    def inflated(self, x0, y0, x1, y1, radius):
        """Boolean grid (iy, ix) over the window, True within `radius` of a return."""
        r = self.res
        i0, j0 = int(math.floor(x0 / r)), int(math.floor(y0 / r))
        nx, ny = int(math.ceil((x1 - x0) / r)) + 1, int(math.ceil((y1 - y0) / r)) + 1
        occ = np.zeros((ny, nx), dtype=bool)
        cells = self.occupied()
        if cells:
            pts = np.array(list(cells))
            ix, iy = pts[:, 0] - i0, pts[:, 1] - j0
            m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            occ[iy[m], ix[m]] = True
        k = int(math.ceil(radius / r))
        out = occ.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dx * dx + dy * dy > k * k or (dx == 0 and dy == 0):
                    continue
                src = occ[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
                out[max(0, dy):ny - max(0, -dy), max(0, dx):nx - max(0, -dx)] |= src
        return Grid(out, i0 * r, j0 * r, r)


class Grid:
    def __init__(self, occ, x0, y0, res):
        self.occ, self.x0, self.y0, self.res = occ, x0, y0, res
        self.ny, self.nx = occ.shape
        self.rows = occ.tolist()

    def hit(self, x, y):
        ix, iy = int((x - self.x0) // self.res), int((y - self.y0) // self.res)
        return 0 <= ix < self.nx and 0 <= iy < self.ny and self.rows[iy][ix]

    def blocked(self, xs, ys):
        return any(self.hit(x, y) for x, y in zip(xs, ys))


class Footprint:
    """Circle centres along the body, measured forward from the rear axle."""

    def __init__(self, rear=0.07, front=0.43, half_width=0.10, margin=0.04):
        n = 3
        self.offsets = [float(v) for v in np.linspace(-rear + half_width, front - half_width, n)]
        self.radius = math.hypot(half_width, (front + rear) / (2 * n)) + margin

    def centres(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return [x + o * c for o in self.offsets], [y + o * s for o in self.offsets]


def free_goal(goal, grid, footprint, shifts=(0.1, 0.2, 0.3)):
    """The goal, or the nearest pose shifted across/along its axis that is clear (an
    exit point seen ~0.3 m off put the goal inside a 36\" connector's inflated wall)."""
    def clear(x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return not any(grid.hit(x + o * c, y + o * s) for o in footprint.offsets)

    gx, gy, gyaw = goal
    if clear(gx, gy, gyaw):
        return goal
    c, s = math.cos(gyaw), math.sin(gyaw)
    for d in shifts:
        for lat, lon in ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)):
            x, y = gx - s * lat + c * lon, gy + c * lat + s * lon
            if clear(x, y, gyaw):
                return (x, y, gyaw)
    return None


def plan(start, goal, grid, footprint, radius, step=0.3, xy_res=0.15, yaw_bins=36,
         tol_xy=0.25, tol_yaw=0.35, reverse_cost=1.6, switch_cost=1.0, turn_cost=0.05,
         max_iter=40000, allow_reverse=True):
    """Hybrid-A* from start to goal (rear-axle poses). Returns [(x, y, yaw, dir)] or None."""
    def collides(x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return any(grid.hit(x + o * c, y + o * s) for o in footprint.offsets)

    gx, gy, gyaw = goal
    if collides(gx, gy, gyaw):
        return None
    # Starting inside the safety margin (e.g. after a slip): let the first moves out.
    escape = 0.35 if collides(*start) else 0.0
    prims = []
    for d in ((1, -1) if allow_reverse else (1,)):
        for k in (-1.0 / radius, 0.0, 1.0 / radius):
            prims.append((d, k))

    def key(x, y, yaw):
        return (int(math.floor(x / xy_res)), int(math.floor(y / xy_res)),
                int(math.floor((_wrap(yaw) + math.pi) / (2 * math.pi) * yaw_bins)) % yaw_bins)

    def h(x, y, yaw):
        return math.hypot(gx - x, gy - y) + 0.5 * radius * abs(_wrap(gyaw - yaw))

    nodes = [(start[0], start[1], start[2], 0, -1, 0.0)]   # x, y, yaw, dir, parent, k
    openq = [(h(*start), 0.0, 0)]
    best_g = {key(*start): 0.0}
    for _ in range(max_iter):
        if not openq:
            return None
        _, g, i = heapq.heappop(openq)
        x, y, yaw, d0, _, k0 = nodes[i]
        if math.hypot(gx - x, gy - y) <= tol_xy and abs(_wrap(gyaw - yaw)) <= tol_yaw:
            path = []
            while i >= 0:
                x, y, yaw, d, i, _ = nodes[i]
                path.append((x, y, yaw, d))
            path.reverse()
            if len(path) > 1:
                path[0] = (*path[0][:3], path[1][3])
            return path
        for d, k in prims:
            ok = True
            px, py, pyaw = x, y, yaw
            for frac in (0.5, 1.0):
                s = d * step * frac
                if k == 0.0:
                    nx_, ny_, nyaw = x + s * math.cos(yaw), y + s * math.sin(yaw), yaw
                else:
                    nyaw = yaw + s * k
                    nx_ = x + (math.sin(nyaw) - math.sin(yaw)) / k
                    ny_ = y - (math.cos(nyaw) - math.cos(yaw)) / k
                if collides(nx_, ny_, nyaw) and \
                        math.hypot(nx_ - start[0], ny_ - start[1]) > escape:
                    ok = False
                    break
                px, py, pyaw = nx_, ny_, nyaw
            if not ok:
                continue
            cost = step * (reverse_cost if d < 0 else 1.0)
            if d0 != 0 and d != d0:
                cost += switch_cost
            if k != k0:
                cost += turn_cost
            ng = g + cost
            kk = key(px, py, pyaw)
            if ng >= best_g.get(kk, math.inf):
                continue
            best_g[kk] = ng
            nodes.append((px, py, pyaw, d, i, k))
            heapq.heappush(openq, (ng + h(px, py, pyaw), ng, len(nodes) - 1))
    return None


def path_blocked(path, grid, footprint, from_index=0):
    for x, y, yaw, _ in path[from_index:]:
        cx, cy = footprint.centres(x, y, yaw)
        if grid.blocked(cx, cy):
            return True
    return False
