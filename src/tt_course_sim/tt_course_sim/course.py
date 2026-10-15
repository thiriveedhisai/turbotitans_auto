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

    def __init__(self, rails=True, seed=0, gap_b=None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.gap_b_override = gap_b
        self.rails = rails
        self.x, self.y, self.yaw = 0.0, 0.0, math.pi       # start heading west
        self.zone_names = []
        self.surf = []           # rows: x, y, top, bot, gx, gy, vib, zone, layer
        self.obst = []           # rows: x, y, bot, top
        self.lines = []          # for plotting: (x1, y1, x2, y2, kind)
        self.discs = []          # (x, y, r, kind)
        self.markers = []        # (zone, x, y)
        self.marker_yaws = []    # heading of travel when the marker is approached
        self.segs3d = []         # (x1, y1, b1, t1, x2, y2, b2, t2, kind) wall/rail segments
        self.slabs3d = []        # (cx, cy, cz, length, width, thick, yaw, pitch, roll, zone)
        self.rough3d = []        # (cx, cy, yaw, length, width, z, vib)
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
            self.segs3d.append((x1, y1, bot, bot + h, x2, y2, bot, bot + h, kind))
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

    def marker(self, zone, width):
        # Zone markers stand on the boundary line just outside the right-hand wall.
        self.marker_at(zone, 0.0, -width / 2)

    def marker_at(self, zone, u, v, base=None, yaw=None):
        self.markers.append((zone,) + self.to_world(u, v, base))
        self.marker_yaws.append(yaw if yaw is not None else (base[2] if base else self.yaw))

    def slab(self, u0, u1, v0, v1, z0, z1, zone, thick=0.08, base=None, roll=0.0):
        """3D record of a driveable rectangle rising z0 -> z1 along u."""
        bx, by, byaw = base if base else (self.x, self.y, self.yaw)
        length = math.hypot(u1 - u0, z1 - z0)
        pitch = -math.atan2(z1 - z0, u1 - u0)
        cx, cy = self.to_world((u0 + u1) / 2, (v0 + v1) / 2, (bx, by, byaw))
        cz = (z0 + z1) / 2 - thick / 2 * math.cos(pitch)
        self.slabs3d.append((cx, cy, cz, length, v1 - v0, thick, byaw, pitch, roll, zone))

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
            self.marker(zone, width)
        self.patch(0, length, -width / 2, width / 2, zone, z=lambda u, v: z0 + slope * u,
                   grad=(slope, 0.0), bot=bot, vib=vib, layer=layer)
        if z0 > 0.005 or z1 > 0.005:
            self.slab(0, length, -width / 2, width / 2, z0, z1, zone)
        if vib > 0:
            cx, cy = self.to_world(length / 2, 0.0)
            self.rough3d.append((cx, cy, self.yaw, length, width, z0, vib))
        if walls[0]:
            self.wall([(0, width / 2), (length, width / 2)], h=wall_h)
        if walls[1]:
            self.wall([(0, -width / 2), (length, -width / 2)], h=wall_h)
        if self.rails and (z0 > 0.05 or z1 > 0.05):
            self.rail_edges(lambda s: (self.to_world(s, 0.0), self.yaw), length, width,
                            lambda s: z0 + slope * s)
        self.advance(length)

    def funnel(self, length, w0, w1, zone, marker=True):
        """Flat lane whose walls taper from width w0 to w1."""
        if marker:
            self.marker(zone, w0)
        w = max(w0, w1)
        self.patch(0, length, -w / 2, w / 2, zone)
        self.wall([(0, w0 / 2), (length, w1 / 2)])
        self.wall([(0, -w0 / 2), (length, -w1 / 2)])
        self.advance(length)

    def step_walls(self, w0, w1):
        """Close the wall step where a lane changes width."""
        self.wall([(0, w0 / 2), (0, w1 / 2)])
        self.wall([(0, -w0 / 2), (0, -w1 / 2)])

    def arc(self, radius, degrees, direction, width, zone, walls=(True, True), z0=0.0,
            z1=None, bot=None, marker=True):
        z1 = z0 if z1 is None else z1
        length = radius * math.radians(degrees)
        slope = (z1 - z0) / length
        cx, cy = self.to_world(0.0, direction * radius)
        yaw0 = self.yaw
        zi = self.zid(zone)
        if marker:
            self.marker(zone, width)

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
        if z0 > 0.005 or z1 > 0.005:
            # Chord segments every ~5 deg, slightly overlapping, for the 3D world.
            n = max(4, int(degrees / 5))
            seg = length / n
            for k in range(n):
                (px, py), hx = centre((k + 0.5) * seg)
                za, zb = z0 + slope * k * seg, z0 + slope * (k + 1) * seg
                self.slab(-seg / 2 - 0.01, seg / 2 + 0.01, -width / 2, width / 2, za, zb, zone,
                          base=(px, py, hx))
        for side, on in ((1, walls[0]), (-1, walls[1])):
            if on:
                prev = None
                for s in np.arange(0, length + 1e-6, 0.05):
                    (px, py), hx = centre(s)
                    pt = (px - math.sin(hx) * side * width / 2, py + math.cos(hx) * side * width / 2)
                    # Walls stand on the (sloped) lane surface, e.g. the helix.
                    zb = max(0.0, z0 + slope * s - 0.03)
                    if prev:
                        self.lines.append(prev[:2] + pt + ('bale',))
                        self.segs3d.append((*prev[:2], prev[2], prev[2] + BALE_H,
                                            *pt, zb, zb + BALE_H, 'bale'))
                        for t in np.linspace(0, 1, 4):
                            self.obst.append((prev[0] + t * (pt[0] - prev[0]),
                                              prev[1] + t * (pt[1] - prev[1]), zb, zb + BALE_H))
                    prev = (*pt, zb)
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
                    self.lines.append(prev[:2] + (x, y, 'rail'))
                    self.segs3d.append((prev[0], prev[1], prev[2], prev[2] + h,
                                        x, y, zfun(s), zfun(s) + h, 'rail'))
                if int(s / 0.02) % 5 == 0:
                    prev = (x, y, zfun(s))

    # ------------------------------------------------------------ the course
    # Geometry follows the official drawing (docs/course, config/track.yaml).
    # World frame: start/finish line at x=0, start lane heading west (-x), north = +y.
    def build(self):
        self.start_pose = (0.35, 0.0, math.pi)          # behind the start line
        self.straight(3.33, 0.813, 'start')
        self.straight(3.25, 0.813, 'ramp_up', walls=(False, False), z0=0.0, z1=0.635)
        # The bridge is the roof of the N-S tunnel; a flat landing follows before the helix.
        self.straight(0.813, 0.813, 'bridge', walls=(False, False), z0=0.635, bot=0.555, layer=1,
                      marker=False)
        self.straight(0.79, 1.054, 'bridge', walls=(False, False), z0=0.635, bot='slab',
                      marker=False)
        self.arc(1.22, 270, +1, 1.054, 'helix', z0=0.635, z1=0.0,
                 bot='slab', marker=False)
        # Helix exit -> 32" funnel -> 5 ft tunnel under the bridge -> 10 ft straight.
        self.funnel(0.81, 1.054, 0.813, 'tunnel')
        self.tunnel_roof(1.524, 0.813)
        self.straight(1.524, 0.813, 'tunnel', wall_h=0.55, marker=False)
        self.wp('tunnel_mid')
        self.straight(3.048, 0.813, 'tunnel', marker=False)
        # Narrow: 32" -> 20" straight, then a 28" right curve on a 6 ft radius.
        self.wp('narrow', back=1.0)
        self.funnel(0.3, 0.813, 0.508, 'narrow')
        self.straight(3.6, 0.508, 'narrow', marker=False)
        self.arc(1.829, 90, -1, 0.711, 'narrow', marker=False)
        self.step_walls(0.711, 0.813)
        self.straight(0.95, 0.813, 'narrow', marker=False)
        # Gravel: 2" ramp up (32"), 4x8 ft gravel box (48"), 2" ramp down (48"), flat.
        self.wp('gravel', back=0.3)
        self.straight(0.813, 0.813, 'gravel', z0=0.0, z1=0.051)
        self.step_walls(0.813, 1.219)
        self.straight(2.438, 1.219, 'gravel', z0=0.051, vib=2.5, marker=False)
        self.straight(0.813, 1.219, 'gravel', z0=0.051, z1=0.0, marker=False)
        self.straight(0.95, 1.219, 'gravel', marker=False)
        self.wp('bank_approach')
        self.build_bank()
        # Pothole: 1.5" ramp up (32"), 8 ft pothole box (48"), 1.5" ramp down (48").
        self.straight(0.813, 0.813, 'pothole', z0=0.0, z1=0.038)
        self.step_walls(0.813, 1.219)
        self.straight(2.438, 1.219, 'pothole', z0=0.038, vib=4.5, marker=False)
        self.straight(0.813, 1.219, 'pothole', z0=0.038, z1=0.0, marker=False)
        self.wp('wide', back=0.3)
        self.build_wide_buckets_hoops()

    def tunnel_roof(self, length, width):
        """3-D only: closed top over the tunnel (the bridge deck covers its south end)."""
        cx, cy = self.to_world(length / 2, 0.0)
        self.slabs3d.append((cx, cy, 0.60, length, width + 0.3, 0.06, self.yaw, 0.0, 0.0, 'tunnel'))

    def build_bank(self):
        """48" x 128" plane rising 8.5 deg away from two diverging lanes split by a divider.
        Local frame B: u = east along the entry lane from the zone line, v = north."""
        B = (self.x, self.y, self.yaw)
        L, depth = 2.6, 1.219
        tan = math.tan(math.radians(8.5))
        self.marker_at('bank_approach', 0.0, -0.66, B)
        self.patch(0.0, L, -0.45, 1.15, 'bank_approach', base=B)
        self.patch(0.0, L, -2.3, -0.45, 'bank_uturn', base=B)
        self.wall([(0.0, 0.61), (L, 1.13)], base=B)
        self.wall([(0.0, -0.61), (L, -0.2)], base=B)
        self.wall([(0.0, -1.09), (L, -0.66)], base=B)
        self.disc(L, -0.43, 0.23, BALE_H, 'post', base=B)
        self.wall([(0.0, -1.90), (L, -2.27)], base=B)
        self.patch(L, L + depth, -2.17, 1.08, 'bank_uturn', z=lambda u, v: (u - L) * tan,
                   grad=(tan, 0.0), base=B)
        self.slab(L, L + depth, -2.17, 1.08, 0.0, depth * tan, 'bank_uturn', thick=0.25, base=B)
        self.bank_box = (*self.to_world(L + depth / 2, -0.545, B), B[2], depth / 2, 1.625)
        self.waypoints['bank_uturn'] = (*self.to_world(L - 1.0, 0.4, B), B[2])
        # Return lane: 32" at the zone line, heading west.
        self.x, self.y = self.to_world(0.0, -1.495, B)
        self.yaw = B[2] + math.pi

    def build_wide_buckets_hoops(self):
        """Fixed world coordinates scaled off the drawing; the pothole lane ends at ~(-3.3, 8.4)."""
        W = (0.0, 0.0, 0.0)
        rng = self.rng
        px, py = self.x, self.y
        # ---- Wide section, entered heading west at its NE corner; turn left (south).
        self.marker_at('wide', px, py + 0.66, W, yaw=math.pi)
        west, south, east = -6.3, 0.85, -2.96
        self.patch(west, px, south, 9.3, 'wide', base=W)
        # Ends where it meets the narrow curve's inner wall (x=-6.3 at y~8.96); running on
        # to 9.27 put a bale stub across the curve lane that wedged every 3-D run.
        self.wall([(west, south), (west, 8.9)], base=W)
        self.wall([(west, south), (east, south)], base=W)
        # Changeable boundaries: zig-zag decoy under the pothole lane, west-wall stubs,
        # an inverted-V decoy on the south wall.
        self.wall([(px, py - 0.61), (east, 7.5), (east - 0.65, 7.0), (east, 6.5),
                   (east - 0.65, 6.0), (east, 5.5), (east, 4.98)], base=W)
        s1 = 7.0 if self.seed == 0 else float(rng.uniform(6.0, 7.6))
        s2 = 4.4 if self.seed == 0 else float(rng.uniform(3.0, 5.0))
        self.wall([(west, s1), (west + 0.8, s1)], base=W)
        self.wall([(west, s2), (west + 0.6, s2 + 0.2)], base=W)
        vx = -4.45 if self.seed == 0 else float(rng.uniform(-5.5, -4.0))
        self.wall([(vx - 0.45, south), (vx, south + 0.8), (vx + 0.45, south)], base=W)

        # ---- Bucket field: local a = east from its west wall, b = north from its south wall.
        depth, span = 3.67, 4.13
        F = (east, south, 0.0)
        hoop_y0, hoop_y1 = 1.36, 3.19
        # The entrance can be anywhere on the wall shared with the wide area (36" gap).
        gap_b = 0.6 if self.seed == 0 else float(rng.uniform(0.46, span - 0.46))
        if self.gap_b_override is not None:
            gap_b = float(self.gap_b_override)
        exit_lo, exit_hi = hoop_y0 - south + 0.46, hoop_y1 - south - 0.46
        exit_b = 1.42 if self.seed == 0 else float(rng.uniform(exit_lo, exit_hi))
        self.gap_center = gap_b
        self.exit_center = exit_b
        self.patch(0.0, depth, 0.0, span, 'buckets', base=F)
        self.wall([(0.0, span), (0.0, gap_b + 0.45)], base=F)
        self.wall([(0.0, gap_b - 0.45), (0.0, 0.0)], base=F)
        self.wall([(0.0, span), (depth, span), (depth, exit_b + 0.457)], base=F)
        self.wall([(depth, exit_b - 0.457), (depth, 0.0)], base=F)
        self.marker_at('buckets', 0.0, gap_b - 0.5 if gap_b >= 0.6 else gap_b + 0.5, F)
        self.waypoints['buckets'] = (*self.to_world(0.1, gap_b, F), 0.0)
        self.buckets = self.place_buckets(gap_b, exit_b, depth, span)
        for a, b in self.buckets:
            self.disc(a, b, 0.15, 0.37, 'bucket', base=F)
        # False obstruction against the wide area's left (bucket-field) wall, clear of the
        # entrance; own RNG so existing layouts don't shift.
        self.obstruction = None
        if self.seed != 0:
            orng = np.random.default_rng(self.seed + 1000)
            gy = south + gap_b
            spots = [y for y in np.arange(south + 0.3, 4.9, 0.05) if abs(y - gy) > 0.9]
            if spots:
                oy = float(orng.choice(spots))
                self.obstruction = (east - 0.2, oy)
                self.disc(east - 0.2, oy, 0.2, BALE_H, 'post', base=W)

        # ---- 36" connector, then the 72" hoop lane heading east.
        ex, ey = self.to_world(depth, exit_b, F)
        lane_x0 = ex + 0.83
        self.patch(ex, lane_x0, ey - 0.457, ey + 0.457, 'hoops', base=W)
        self.wall([(ex, ey + 0.457), (lane_x0, ey + 0.457), (lane_x0, hoop_y1)], base=W)
        self.wall([(ex, ey - 0.457), (lane_x0, ey - 0.457), (lane_x0, hoop_y0)], base=W)
        self.marker_at('hoops', ex, ey - 0.5, W, yaw=0.0)
        self.waypoints['hoops'] = (ex - 0.3, ey, 0.0)
        ux, uy, ur = 7.1, 0.31, 2.88                   # U-turn outer wall
        tip = (7.57, -0.12)                            # divider tip
        self.patch(lane_x0, ux, hoop_y0, hoop_y1, 'hoops', base=W)
        self.patch(ux - 2.0, ux + ur, uy - ur, hoop_y1, 'hoops', base=W)
        self.wall([(lane_x0, hoop_y1), (ux, hoop_y1)], base=W)
        self.wall([(ux + ur * math.cos(t), uy + ur * math.sin(t))
                   for t in np.linspace(math.atan2(hoop_y1 - uy, 0.0), -math.pi / 2, 30)], base=W)
        self.wall([(lane_x0, hoop_y0), (5.2, hoop_y0), tip, (5.2, 0.61), (0.85, 0.61), (0.0, 0.405)],
                  base=W)
        self.wall([(ux, uy - ur), (5.0, -0.61), (0.85, -0.61), (0.0, -0.405)], base=W)
        self.hoops = []
        self.hoop_legs = []
        for hx in (lane_x0 + 2.438, lane_x0 + 4.876):
            w = 0.635 if self.seed == 0 else float(rng.uniform(0.51, 0.76))
            hy = float(rng.uniform(hoop_y0 + w / 2 + 0.1, hoop_y1 - w / 2 - 0.1))
            self.disc(hx, hy + w / 2, 0.025, 0.9, 'hoop', base=W)
            self.disc(hx, hy - w / 2, 0.025, 0.9, 'hoop', base=W)
            self.hoops.append((hx, hy))
            self.hoop_legs.append(((hx, hy - w / 2), (hx, hy + w / 2)))
        w = 0.635 if self.seed == 0 else float(rng.uniform(0.51, 0.76))
        x3 = float(rng.uniform(tip[0] + 0.25 + w / 2, ux + ur - w / 2 - 0.15))
        self.disc(x3 - w / 2, tip[1], 0.025, 0.9, 'hoop', base=W)
        self.disc(x3 + w / 2, tip[1], 0.025, 0.9, 'hoop', base=W)
        self.hoops.append((x3, tip[1]))
        self.hoop_legs.append(((x3 - w / 2, tip[1]), (x3 + w / 2, tip[1])))

        # ---- Car wash (48", ribbons) heading west, 48" -> 32" funnel to the finish line.
        self.patch(0.0, 5.2, -0.61, 0.61, 'car_wash', base=W)
        self.marker_at('car_wash', 4.97, 0.66, W, yaw=math.pi)
        self.waypoints['car_wash'] = (5.4, -0.1, math.pi)
        self.noise_boxes.append((2.885, 0.0, math.pi, 1.12, 0.61))
        self.finish = (0.0, 0.0, math.pi, 0.15, 0.5)
        self.marker_at('finish', 0.85, 0.66, W, yaw=math.pi)

    # ------------------------------------------------------------ randomisation
    def place_buckets(self, gap_b, exit_b, depth, span, clear=0.35):
        """2-9 buckets (a = depth into field, b = across) with a guaranteed path
        at least `clear` wide from the entrance gap to the exit. Seed 0 = drawing."""
        if self.seed == 0:
            return [(0.84, 3.18), (1.79, 3.18), (2.73, 3.18), (2.91, 1.85), (0.85, 1.30), (2.40, 1.17)]
        need = clear / 2 + 0.15 + 0.10      # half path + bucket radius + truck half width
        for _ in range(500):
            n = int(self.rng.integers(2, 10))
            pts = [(float(self.rng.uniform(0.5, depth - 0.4)), float(self.rng.uniform(0.3, span - 0.3)))
                   for _ in range(n)]
            if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.45 for i, p in enumerate(pts) for q in pts[:i]):
                continue
            # A car can't sidestep at the mouths: keep the entrance and exit approaches open.
            if any(math.hypot(a, b - gap_b) < 0.9 or math.hypot(depth - a, b - exit_b) < 0.9
                   for a, b in pts):
                continue
            if self._path_exists(pts, gap_b, (exit_b - 0.457, exit_b + 0.457), (depth, 0.0, span), need):
                return pts
        return [(1.8, 0.6), (1.8, span - 0.6)]

    @staticmethod
    def _path_exists(pts, gap_c, exit_b, field, need, res=0.05):
        """Grid flood fill for the truck centre from the entrance to the exit."""
        depth, b0, b1 = field
        na, nb = int(depth / res) + 1, int((b1 - b0) / res) + 1
        free = np.ones((na, nb), dtype=bool)
        a = np.arange(na)[:, None] * res
        b = b0 + np.arange(nb)[None, :] * res
        for pa, pb in pts:
            free &= np.hypot(a - pa, b - pb) > need
        free &= (b > b0 + 0.15) & (b < b1 - 0.15)
        start = [(0, j) for j in range(nb) if abs(b0 + j * res - gap_c) < 0.2 and free[0, j]]
        goal = {(na - 1, j) for j in range(nb) if exit_b[0] + 0.15 < b0 + j * res < exit_b[1] - 0.15}
        seen, stack = set(start), list(start)
        while stack:
            i, j = stack.pop()
            if (i, j) in goal:
                return True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                k = (i + di, j + dj)
                if 0 <= k[0] < na and 0 <= k[1] < nb and k not in seen and free[k]:
                    seen.add(k)
                    stack.append(k)
        return False

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
