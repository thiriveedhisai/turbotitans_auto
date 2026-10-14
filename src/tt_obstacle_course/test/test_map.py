import math

import numpy as np

from tt_obstacle_course.map_tools import CourseMap, Localizer, Mapper, compose, relative

SENSOR = (-0.15, 0.0)


def walls_course():
    """An L-shaped 1.2 m lane plus a 4 x 3 m open area with a post (like a bucket field)."""
    segs = [((0, -0.6), (8, -0.6)), ((0, 0.6), (6.8, 0.6)), ((8, -0.6), (8, 4.0)),
            ((6.8, 0.6), (6.8, 1.0)), ((6.8, 1.0), (4.8, 1.0)), ((4.8, 1.0), (4.8, 4.0)),
            ((4.8, 4.0), (8, 4.0)), ((0, -0.6), (0, 0.6))]
    return segs


def post(cx, cy, r=0.15, n=12):
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n + 1)]
    return list(zip(pts[:-1], pts[1:]))


def scan(pose, segs, n=360, max_r=8.0):
    """Points (vehicle frame) seen by a lidar at SENSOR on the vehicle at pose."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    ox, oy = pose[0] + c * SENSOR[0], pose[1] + s * SENSOR[0]
    a = np.linspace(-math.pi, math.pi, n, endpoint=False) + pose[2]
    dx, dy = np.cos(a), np.sin(a)
    best = np.full(n, np.inf)
    for (x1, y1), (x2, y2) in segs:
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((x1 - ox) * ey - (y1 - oy) * ex) / den
            u = ((x1 - ox) * dy - (y1 - oy) * dx) / den
        hit = (np.abs(den) > 1e-9) & (t > 0.05) & (u >= 0) & (u <= 1)
        best = np.where(hit & (t < best), t, best)
    ok = best < max_r
    wx, wy = ox + best[ok] * dx[ok], oy + best[ok] * dy[ok]
    rx, ry = wx - pose[0], wy - pose[1]
    return np.stack([c * rx + s * ry, -s * rx + c * ry], 1)


def path():
    poses = [(x, 0.0, 0.0) for x in np.arange(0.5, 7.4, 0.1)]
    poses += [(7.4, y, math.pi / 2) for y in np.arange(0.1, 3.2, 0.1)]
    return poses


def drifting_odom(poses, yaw_rate=math.radians(0.05), scale=1.04):
    """Odometry that over-counts distance by 4% and drifts 0.05 deg per 10 cm (gyro-like)."""
    out = [poses[0]]
    for a, b in zip(poses, poses[1:]):
        d = relative(a, b)
        out.append(compose(out[-1], (d[0] * scale, d[1] * scale, d[2] + yaw_rate)))
    return out


def test_map_then_localize_on_changed_course():
    segs = walls_course() + post(6.0, 2.5)
    truth = path()
    odom = drifting_odom(truth)
    mapper = Mapper(SENSOR)
    for i, (t, o) in enumerate(zip(truth, odom)):
        mapper.update(float(i), o, scan(t, segs), 'lane' if t[1] < 0.5 else 'field')
    built = [p[1:4] for p in mapper.traj]
    err = max(math.hypot(b[0] - t[0], b[1] - t[1]) for b, t in zip(built, truth))
    odom_err = max(math.hypot(o[0] - t[0], o[1] - t[1]) for o, t in zip(odom, truth))
    assert odom_err > 0.25
    assert err < 0.12, f'mapping drifted {err:.2f} m'

    cmap = CourseMap.from_grid(mapper.grid, mapper.traj)
    assert cmap.zone_at(*truth[-1][:2]) == 'field'
    # Course changed: the post moved 1 m and an extra one stands in the field.
    changed = walls_course() + post(5.4, 2.0) + post(6.6, 3.2)
    loc = Localizer(cmap)
    odom2 = drifting_odom(truth, yaw_rate=-math.radians(0.05), scale=0.96)
    errs = []
    for i, (t, o) in enumerate(zip(truth, odom2)):
        p = loc.update(o, scan(t, changed), float(i))
        errs.append(math.hypot(p[0] - t[0], p[1] - t[1]))
    assert loc.state == 'tracking'
    assert max(errs) < 0.15, f'localization error {max(errs):.2f} m'


def ring():
    """1.2 m lane round a 10 x 6 m block: start and finish are the same place."""
    out = [(0, 0), (10, 0), (10, 6), (0, 6)]
    inn = [(1.2, 1.2), (8.8, 1.2), (8.8, 4.8), (1.2, 4.8)]
    segs = [(a, b) for poly in (out, inn) for a, b in zip(poly, poly[1:] + poly[:1])]
    return segs + post(5.0, 0.35, 0.1) + post(9.6, 3.0, 0.1) + post(5.0, 5.65, 0.1)


def ring_path():
    poses = []
    for x in np.arange(2.0, 9.4, 0.1):
        poses.append((x, 0.6, 0.0))
    for y in np.arange(0.6, 5.4, 0.1):
        poses.append((9.4, y, math.pi / 2))
    for x in np.arange(9.4, 0.6, -0.1):
        poses.append((x, 5.4, math.pi))
    for y in np.arange(5.4, 0.6, -0.1):
        poses.append((0.6, y, -math.pi / 2))
    for x in np.arange(0.6, 2.05, 0.1):
        poses.append((x, 0.6, 0.0))
    return poses


def test_loop_closes_at_the_finish():
    segs = ring()
    truth = ring_path()
    odom = drifting_odom(truth, yaw_rate=math.radians(0.08), scale=1.05)
    m = Mapper(SENSOR)
    for i, (t, o) in enumerate(zip(truth, odom)):
        m.update(float(i), o, scan(t, segs))
    end_before = math.hypot(m.pose[0] - truth[-1][0], m.pose[1] - truth[-1][1])
    res = m.close_loop(scan(truth[-1], segs), min_path=20.0)
    assert res is not None
    end_after = math.hypot(m.pose[0] - truth[-1][0], m.pose[1] - truth[-1][1])
    worst = max(math.hypot(f[1] - t[0], f[2] - t[1]) for f, t in zip(m.traj, truth))
    assert end_after < 0.1, f'end {end_before:.2f} -> {end_after:.2f} m'
    assert worst < max(0.3, end_before), f'worst pose error after closing {worst:.2f} m'


def test_loop_closure_fixes_an_odometry_only_map():
    segs = ring()
    truth = ring_path()
    odom = drifting_odom(truth, yaw_rate=math.radians(0.04), scale=1.03)
    m = Mapper(SENSOR, match=False)
    for i, (t, o) in enumerate(zip(truth, odom)):
        m.update(float(i), o, scan(t, segs))
    before = max(math.hypot(f[1] - t[0], f[2] - t[1]) for f, t in zip(m.traj, truth))
    assert m.close_loop(scan(truth[-1], segs), min_path=20.0) is not None
    after = max(math.hypot(f[1] - t[0], f[2] - t[1]) for f, t in zip(m.traj, truth))
    end = math.hypot(m.pose[0] - truth[-1][0], m.pose[1] - truth[-1][1])
    assert before > 0.8
    assert end < 0.1 and after < before / 2, f'{before:.2f} -> {after:.2f} m, end {end:.2f}'
