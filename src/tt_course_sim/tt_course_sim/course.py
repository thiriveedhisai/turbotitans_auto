"""2.5-D model of the obstacle course, built turtle-style from the course drawing.

Everything is rasterised into grids (4 cm) so the sim can answer, per cell:
driveable surface height/gradient (ramps, bridge, helix, bank), vibration
(gravel, potholes), obstacle height band (hay bales, rails, buckets, hoop legs)
and which zone it belongs to. The lidar is ray-marched through the same grids,
so tilted scans really do hit the floor or see over walls.

Dimensions follow the drawing where given; connecting lengths are estimates.
"""

import math

import numpy as np

RES = 0.02
BALE_H = 0.36
FALL_DROP = 0.20         # metres: a drop larger than this is a fall


class Course:

    def __init__(self, rails=True, seed=0, hoop_width=0.60):
        self.rng = np.random.default_rng(seed)
        self.rails = rails
        self.hoop_width = hoop_width
        self.x, self.y, self.yaw = 0.0, 0.0, math.pi       # start heading west
        self.zone_names = []
        self.surf = []           # rows: x, y, top, bot, gx, gy, vib, zone, layer
        self.obst = []           # rows: x, y, bot, top
        self.lines = []          # for plotting: (x1, y1, x2, y2, kind)
        self.discs = []          # (x, y, r, kind)
        self.markers = []        # (zone, x, y)
        self.noise_boxes = []    # (cx, cy, yaw, half_len, half_wid)
        self.finish = None       # (cx, cy, yaw, half_len, half_wid)
        self.start_pose = None
        self.waypoints = {}
        self.build()
        self.rasterise()

    # ------------------------------------------------------------ helpers
    def zid(self, name):
        if name not in self.zone_names:
            self.zone_names.append(name)
        return self.zone_names.index(name)

    def to_world(self, u, v, base=None):
        bx, by, byaw = base if base else (self.x, self.y, self.yaw)
        c, s = math.cos(byaw), math.sin(byaw)
        return bx + c * u - s * v, by + s * u + c * v

    def wall(self, pts, h=BALE_H, bot=0.0, kind='bale', base=None):
        """Polyline wall from local (u, v) points."""
        world = [self.to_world(u, v, base) for u, v in pts]
        for (x1, y1), (x2, y2) in zip(world, world[1:]):
            self.lines.append((x1, y1, x2, y2, kind))
            n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 0.02))
            for t in np.linspace(0, 1, n):
                self.obst.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1), bot, bot + h))

    def disc(self, u, v, r, h, kind, base=None):
        x, y = self.to_world(u, v, base)
        self.discs.append((x, y, r, kind))
        for dx in np.arange(-r, r + 1e-6, 0.02):
            for dy in np.arange(-r, r + 1e-6, 0.02):
                if dx * dx + dy * dy <= r * r:
                    self.obst.append((x + dx, y + dy, 0.0, h))

    def patch(self, u0, u1, v0, v1, zone, z=lambda u, v: 0.0, grad=(0.0, 0.0),
              bot=None, vib=0.0, layer=0, base=None):
        """Driveable rectangle in local coords. grad is local (dz/du, dz/dv)."""
        byaw = base[2] if base else self.yaw
        c, s = math.cos(byaw), math.sin(byaw)
        gx, gy = c * grad[0] - s * grad[1], s * grad[0] + c * grad[1]
        zi = self.zid(zone)
        for u in np.arange(u0, u1 + 1e-6, RES / 2):
            for v in np.arange(v0, v1 + 1e-6, RES / 2):
                x, y = self.to_world(u, v, base)
                top = z(u, v)
                b = (top - 0.08) if bot == 'slab' else (bot if bot is not None else -10.0)
                self.surf.append((x, y, top, b, gx, gy, vib, zi, layer))

    def marker(self, zone, u=0.0, v=0.0):
        self.markers.append((zone,) + self.to_world(u, v))

    def wp(self, name, back=0.0):
        """Remember the current pose (optionally `back` metres earlier) as a test start."""
        self.waypoints[name] = (*self.to_world(-back, 0.0), self.yaw)

    def waypoint(self, name):
        return self.waypoints[name]

    def advance(self, length):
        self.x, self.y = self.to_world(length, 0.0)

    # ------------------------------------------------------------ pieces
    def straight(self, length, width, zone, walls=(True, True), z0=0.0, z1=None,
                 bot=None, vib=0.0, layer=0, wall_h=BALE_H, marker=True):
        z1 = z0 if z1 is None else z1
        slope = (z1 - z0) / length
        if marker:
            self.marker(zone)
        self.patch(0, length, -width / 2, width / 2, zone, z=lambda u, v: z0 + slope * u,
                   grad=(slope, 0.0), bot=bot, vib=vib, layer=layer)
        if walls[0]:
            self.wall([(0, width / 2), (length, width / 2)], h=wall_h)
        if walls[1]:
            self.wall([(0, -width / 2), (length, -width / 2)], h=wall_h)
        if self.rails and (z0 > 0.05 or z1 > 0.05):
            self.rail_edges(lambda s: (self.to_world(s, 0.0), self.yaw), length, width,
                            lambda s: z0 + slope * s)
        self.advance(length)

    def arc(self, radius, degrees, direction, width, zone, walls=(True, True), z0=0.0,
            z1=None, bot=None, marker=True):
        z1 = z0 if z1 is None else z1
        length = radius * math.radians(degrees)
        slope = (z1 - z0) / length
        cx, cy = self.to_world(0.0, direction * radius)
        yaw0 = self.yaw
        zi = self.zid(zone)
        if marker:
            self.marker(zone)

        def centre(s):
            phi = direction * s / radius
            hx = yaw0 + phi
            # Position relative to centre rotates with the heading.
            ox, oy = -direction * radius * -math.sin(yaw0), -direction * radius * math.cos(yaw0)
            px = cx + ox * math.cos(phi) - oy * math.sin(phi)
            py = cy + ox * math.sin(phi) + oy * math.cos(phi)
            return (px, py), hx

        # Half-cell steps at the outer edge so the rasterised surface has no holes.
        ds = RES / 2 * radius / (radius + width / 2)
        for s in np.arange(0, length + 1e-6, ds):
            (px, py), hx = centre(s)
            top = z0 + slope * s
            b = (top - 0.08) if bot == 'slab' else -10.0
            for t in np.arange(-width / 2, width / 2 + 1e-6, RES / 2):
                x, y = px - math.sin(hx) * t, py + math.cos(hx) * t
                self.surf.append((x, y, top, b, slope * math.cos(hx), slope * math.sin(hx),
                                  0.0, zi, 0))
        for side, on in ((1, walls[0]), (-1, walls[1])):
            if on:
                prev = None
                for s in np.arange(0, length + 1e-6, 0.05):
                    (px, py), hx = centre(s)
                    pt = (px - math.sin(hx) * side * width / 2, py + math.cos(hx) * side * width / 2)
                    if prev:
                        self.lines.append(prev + pt + ('bale',))
                        for t in np.linspace(0, 1, 4):
                            self.obst.append((prev[0] + t * (pt[0] - prev[0]),
                                              prev[1] + t * (pt[1] - prev[1]), 0.0, BALE_H))
                    prev = pt
        if self.rails and (z0 > 0.05 or z1 > 0.05):
            self.rail_edges(centre, length, width, lambda s: z0 + slope * s)
        (self.x, self.y), self.yaw = centre(length)

    def rail_edges(self, centre, length, width, zfun, h=0.15):
        for side in (1, -1):
            prev = None
            for s in np.arange(0, length + 1e-6, 0.02):
                (px, py), hx = centre(s)
                x = px - math.sin(hx) * side * (width / 2 + 0.02)
                y = py + math.cos(hx) * side * (width / 2 + 0.02)
                self.obst.append((x, y, zfun(s), zfun(s) + h))
                if prev and int(s / 0.02) % 5 == 0:
                    self.lines.append(prev + (x, y, 'rail'))
                if int(s / 0.02) % 5 == 0:
                    prev = (x, y)

    # ------------------------------------------------------------ the course
    def build(self):
        # Start (32" lane, heading west)
        self.start_pose = (*self.to_world(0.4, 0.0), self.yaw)
        self.straight(3.3, 0.81, 'start')
        # 20% ramp up to a 25" bridge, then the helix down (rails optional)
        self.straight(3.2, 0.81, 'ramp_up', walls=(False, False), z0=0.0, z1=0.64)
        self.straight(1.8, 0.81, 'bridge', walls=(False, False), z0=0.64, bot=0.56, layer=1)
        self.arc(1.22, 270, +1, 1.054, 'helix', walls=(False, False), z0=0.64, z1=0.0, bot='slab')
        # Tunnel under the bridge, then the 10 ft corridor
        self.straight(1.8, 0.81, 'tunnel', wall_h=0.55)
        self.wp('tunnel_mid')
        self.straight(3.05, 0.81, 'tunnel', marker=False)
        # Narrow: 20" straight then 28" right curve (6 ft radius)
        self.wp('narrow', back=1.0)
        self.straight(2.3 + 1.2, 0.508, 'narrow')
        self.arc(1.83, 90, -1, 0.71, 'narrow', marker=False)
        # Gravel: 2" ramp, 4x8 ft gravel, 2" ramp down, flat
        self.wp('gravel', back=0.3)
        self.straight(0.81, 0.81, 'gravel', z0=0.0, z1=0.05)
        self.straight(2.44, 1.22, 'gravel', z0=0.05, vib=2.5, marker=False)
        self.straight(0.81, 1.22, 'gravel', z0=0.05, z1=0.0, marker=False)
        self.straight(1.0, 1.22, 'gravel', marker=False)
        # Bank approach (flat, funnel simplified to a straight lane)
        self.wp('bank_approach')
        self.straight(4.0, 1.22, 'bank_approach')
        self.wp('bank_uturn')
        self.build_bank()
        # Pothole lane (continues west from the bank return lane)
        self.straight(0.81, 1.22, 'pothole', z0=0.0, z1=0.038)
        self.straight(2.44, 1.22, 'pothole', z0=0.038, vib=4.5, marker=False)
        self.straight(0.81, 1.22, 'pothole', z0=0.038, z1=0.0, marker=False)
        self.wp('wide', back=0.3)
        self.build_wide_buckets_hoops()

    def build_bank(self):
        base = (self.x, self.y, self.yaw)
        tan = math.tan(math.radians(8.5))
        # Open apron + 48" x 128" plane rising away from the lanes; no walls.
        self.patch(0.0, 0.3, -2.4, 0.9, 'bank_uturn', base=base)
        self.patch(0.3, 1.52, -2.39, 0.87, 'bank_uturn', z=lambda u, v: (u - 0.3) * tan,
                   grad=(tan, 0.0), base=base)
        # Divider between entry lane (v=0) and return lane (v=-1.37), post at its end.
        self.wall([(-2.4, -0.61), (0.0, -0.61)], base=base)
        self.wall([(-2.4, -0.76), (0.0, -0.76)], base=base)
        self.disc(0.0, -0.685, 0.15, BALE_H, 'post', base=base)
        self.wall([(-2.4, -1.98), (0.0, -1.98)], base=base)
        self.markers.append(('bank_uturn',) + self.to_world(0.0, 0.0, base))
        # Return lane back toward the wide section.
        self.x, self.y = self.to_world(0.0, -1.37, base)
        self.yaw = base[2] + math.pi
        self.straight(1.0, 1.22, 'pothole', walls=(False, False))

    def build_wide_buckets_hoops(self):
        # ---- Wide section: local u = forward (west), v = left (south)
        A = (self.x, self.y, self.yaw)
        self.markers.append(('wide',) + self.to_world(0.0, 0.0, A))
        self.patch(0.0, 3.35, -0.7, 7.0, 'wide', base=A)
        self.wall([(0.0, -0.61), (0.0, -0.7), (3.35, -0.7), (3.35, 7.0), (-0.0, 7.0)], base=A)
        # East wall with a zig-zag decoy (closed notches) and the real gap at v 5.9-6.8
        self.wall([(0.0, 0.61), (0.0, 1.0), (0.7, 1.5), (0.0, 2.0), (0.7, 2.5), (0.0, 3.0),
                   (0.0, 5.9)], base=A)
        self.wall([(0.0, 6.8), (0.0, 7.0)], base=A)
        # West-wall stubs and the V decoy that opens north
        self.wall([(3.35, 2.0), (2.6, 2.0)], base=A)
        self.wall([(3.35, 4.0), (2.7, 4.3)], base=A)
        self.wall([(0.65, 6.2), (1.35, 7.0), (2.6, 6.2)], base=A)

        # ---- Bucket field east of the gap: local a = east (-u), b = v
        self.markers.append(('buckets',) + self.to_world(0.0, 6.35, A))
        self.waypoints['buckets'] = (*self.to_world(-0.1, 6.35, A), A[2] + math.pi)
        self.patch(-3.6, 0.0, 3.4, 7.0, 'buckets', base=A)
        self.wall([(0.0, 3.4), (-3.6, 3.4), (-3.6, 5.5)], base=A)
        self.wall([(-3.6, 6.4), (-3.6, 7.0), (0.0, 7.0)], base=A)
        for a, b in ((0.9, 4.0), (2.1, 4.0), (3.0, 4.1), (1.2, 6.5), (2.3, 5.4), (2.9, 6.6)):
            self.disc(-a, b, 0.15, 0.37, 'bucket', base=A)
        # 36" connector east to the hoop lane
        self.patch(-4.6, -3.6, 5.5, 6.4, 'hoops', base=A)
        self.wall([(-3.6, 5.5), (-4.6, 5.5)], base=A)
        self.wall([(-3.6, 6.4), (-4.6, 6.4)], base=A)

        # ---- Hoops: local x = east along the lane, y = left (north)
        hx, hy = self.to_world(-4.6, 5.95, A)
        H = (hx, hy, A[2] + math.pi)
        self.waypoints['hoops'] = (*self.to_world(-0.6, 0.0, H), H[2])
        self.markers.append(('hoops',) + self.to_world(0.0, 0.0, H))
        self.patch(0.0, 5.3, -0.915, 0.915, 'hoops', base=H)
        self.wall([(0.0, 0.455), (0.0, 0.915), (5.3, 0.915)], base=H)
        self.wall([(0.0, -0.455), (0.0, -0.915), (5.3, -0.915), (3.8, -2.69), (-3.2, -2.69)],
                  base=H)
        # U-turn: outer arc centred (5.3, -1.8), r = 2.72
        cx, cy, r = 5.3, -1.8, 2.72
        arc_pts = [(cx + r * math.cos(t), cy + r * math.sin(t))
                   for t in np.linspace(math.pi / 2, -math.pi / 2, 25)]
        self.wall(arc_pts + [(4.6, -3.91), (-3.2, -3.91)], base=H)
        self.patch(5.3, 8.02, -4.52, 0.915, 'hoops', base=H)
        self.patch(-3.2, 5.3, -3.91, -2.69, 'car_wash', base=H)
        hw = self.hoop_width / 2
        for x in (2.44, 4.88):
            y = float(self.rng.uniform(-0.55, 0.55))
            self.disc(x, y + hw, 0.025, 0.9, 'hoop', base=H)
            self.disc(x, y - hw, 0.025, 0.9, 'hoop', base=H)
        x3 = float(self.rng.uniform(5.9, 7.3))
        self.disc(x3 - hw, -0.915, 0.025, 0.9, 'hoop', base=H)
        self.disc(x3 + hw, -0.915, 0.025, 0.9, 'hoop', base=H)
        self.hoops = [self.to_world(2.44, 0, H), self.to_world(4.88, 0, H), self.to_world(x3, -0.915, H)]

        # ---- Car wash (ribbons) and finish, heading west along the lane
        self.markers.append(('car_wash',) + self.to_world(2.4, -3.3, H))
        cwx, cwy = self.to_world(1.0, -3.3, H)
        self.noise_boxes.append((cwx, cwy, H[2], 1.2, 0.61))
        fx, fy = self.to_world(-3.0, -3.3, H)
        self.finish = (fx, fy, H[2], 0.2, 0.61)
        self.markers.append(('finish',) + self.to_world(-2.0, -3.3, H))

    # ------------------------------------------------------------ grids
    def rasterise(self):
        surf = np.array(self.surf, dtype=float)
        obst = np.array(self.obst, dtype=float)
        allx = np.concatenate([surf[:, 0], obst[:, 0]])
        ally = np.concatenate([surf[:, 1], obst[:, 1]])
        self.x0, self.y0 = allx.min() - 2.0, ally.min() - 2.0
        self.nx = int((allx.max() + 2.0 - self.x0) / RES) + 1
        self.ny = int((ally.max() + 2.0 - self.y0) / RES) + 1
        shape = (self.ny, self.nx)
        self.top = np.zeros(shape)
        self.bot = np.zeros(shape)
        self.gx = np.zeros(shape)
        self.gy = np.zeros(shape)
        self.vib = np.zeros(shape)
        self.zone = np.full(shape, -1, dtype=int)
        self.deck = np.full(shape, np.nan)
        self.deck_bot = np.full(shape, np.nan)
        self.o_bot = np.full(shape, np.inf)
        self.o_top = np.full(shape, -np.inf)

        g = surf[surf[:, 8] == 0]
        # Where pieces overlap, the higher surface wins (ramps over floors).
        g = g[np.argsort(g[:, 2], kind='stable')]
        i, j = self.cell(g[:, 0], g[:, 1])
        self.top[i, j], self.bot[i, j] = g[:, 2], g[:, 3]
        self.gx[i, j], self.gy[i, j], self.vib[i, j] = g[:, 4], g[:, 5], g[:, 6]
        self.zone[i, j] = g[:, 7].astype(int)
        d = surf[surf[:, 8] == 1]
        i, j = self.cell(d[:, 0], d[:, 1])
        self.deck[i, j], self.deck_bot[i, j] = d[:, 2], d[:, 3]
        self.deck_zone = np.full(shape, -1, dtype=int)
        self.deck_zone[i, j] = d[:, 7].astype(int)

        # Collisions use exact cells; the lidar grid below is thickened.
        self.c_bot = np.full(shape, np.inf)
        self.c_top = np.full(shape, -np.inf)
        i, j = self.cell(obst[:, 0], obst[:, 1])
        np.minimum.at(self.c_bot, (i, j), obst[:, 2])
        np.maximum.at(self.c_top, (i, j), obst[:, 3])
        # Lidar grid: +1 cell cross so 2 cm ray steps can't slip through diagonals.
        for di, dj in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            i, j = self.cell(obst[:, 0], obst[:, 1])
            i, j = np.clip(i + di, 0, self.ny - 1), np.clip(j + dj, 0, self.nx - 1)
            np.minimum.at(self.o_bot, (i, j), obst[:, 2])
            np.maximum.at(self.o_top, (i, j), obst[:, 3])

    def cell(self, x, y):
        j = np.clip(((np.asarray(x) - self.x0) / RES).astype(int), 0, self.nx - 1)
        i = np.clip(((np.asarray(y) - self.y0) / RES).astype(int), 0, self.ny - 1)
        return i, j

    # ------------------------------------------------------------ queries
    def ground(self, x, y, z_prev):
        """(z, gx, gy, vib, zone_name) of the surface under (x, y)."""
        i, j = self.cell(x, y)
        i, j = int(i), int(j)
        if not np.isnan(self.deck[i, j]) and z_prev > self.deck[i, j] - 0.2:
            return float(self.deck[i, j]), 0.0, 0.0, 0.0, self.name(self.deck_zone[i, j])
        return (float(self.top[i, j]), float(self.gx[i, j]), float(self.gy[i, j]),
                float(self.vib[i, j]), self.name(self.zone[i, j]))

    def name(self, zi):
        return self.zone_names[zi] if zi >= 0 else ''

    def blocked(self, xs, ys, z):
        """True if any footprint point hits an obstacle band overlapping the body."""
        i, j = self.cell(xs, ys)
        return bool(np.any((self.c_bot[i, j] < z + 0.25) & (self.c_top[i, j] > z + 0.02)))

    def in_box(self, box, x, y):
        cx, cy, yaw, hl, hw = box
        dx, dy = x - cx, y - cy
        u = dx * math.cos(yaw) + dy * math.sin(yaw)
        v = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return abs(u) <= hl and abs(v) <= hw

    def scan(self, lx, ly, lz, yaw, nose_up, roll_left, n=360, rmax=8.0, step=0.02,
             z_robot=0.0):
        """Ray-march n beams. Returns ranges (inf = no return)."""
        a = -math.pi + np.arange(n) * (2 * math.pi / n)
        r = np.arange(0.12, rmax, step)
        tan_e = math.tan(nose_up) * np.cos(a) + math.tan(roll_left) * np.sin(a)
        ang = yaw + a
        px = lx + np.outer(np.cos(ang), r)
        py = ly + np.outer(np.sin(ang), r)
        pz = lz + np.outer(tan_e, r)
        i, j = self.cell(px, py)
        surf = (pz <= self.top[i, j]) & (pz >= self.bot[i, j])
        obst = (pz >= self.o_bot[i, j]) & (pz <= self.o_top[i, j])
        deck = (pz >= self.deck_bot[i, j]) & (pz <= self.deck[i, j])
        hit = surf | obst | deck | (pz <= 0.0)
        first = np.argmax(hit, axis=1)
        return np.where(hit.any(axis=1), r[first], np.inf)


def preview(path, course=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = course or Course()
    fig, ax = plt.subplots(figsize=(14, 11))
    elev = np.where(c.top > 0.01, c.top, np.nan)
    ax.imshow(elev, origin='lower', extent=(c.x0, c.x0 + c.nx * RES, c.y0, c.y0 + c.ny * RES),
              cmap='Oranges', alpha=0.6)
    vib = np.where(c.vib > 0, c.vib, np.nan)
    ax.imshow(vib, origin='lower', extent=(c.x0, c.x0 + c.nx * RES, c.y0, c.y0 + c.ny * RES),
              cmap='Greys', alpha=0.4)
    deck = np.where(~np.isnan(c.deck), 1.0, np.nan)
    ax.imshow(deck, origin='lower', extent=(c.x0, c.x0 + c.nx * RES, c.y0, c.y0 + c.ny * RES),
              cmap='Blues', alpha=0.4, vmin=0, vmax=2)
    colours = {'bale': '#c8a020', 'rail': '#8080ff'}
    for x1, y1, x2, y2, kind in c.lines:
        ax.plot([x1, x2], [y1, y2], color=colours.get(kind, 'k'), lw=3 if kind == 'bale' else 1)
    for x, y, r, kind in c.discs:
        ax.add_patch(plt.Circle((x, y), max(r, 0.05), color={'bucket': 'orange', 'hoop': 'red',
                                                             'post': 'brown'}[kind]))
    for zone, x, y in c.markers:
        ax.plot(x, y, 'k^', ms=6)
        ax.annotate(zone, (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points')
    sx, sy, syaw = c.start_pose
    ax.arrow(sx, sy, 0.6 * math.cos(syaw), 0.6 * math.sin(syaw), width=0.08, color='green')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title('Simulated obstacle course (orange = elevated, grey = rough, blue = bridge)')
    return fig, ax


if __name__ == '__main__':
    import sys
    fig, _ = preview(sys.argv[1] if len(sys.argv) > 1 else 'course.png')
    fig.savefig(sys.argv[1] if len(sys.argv) > 1 else 'course.png', dpi=80, bbox_inches='tight')
