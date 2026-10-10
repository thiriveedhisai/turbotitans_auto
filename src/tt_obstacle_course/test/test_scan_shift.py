"""scan_shift: lidar travel estimate used to dead-reckon ZED dropouts."""

import math

import numpy as np

from tt_obstacle_course.scan_utils import scan_shift

ANGLES = np.linspace(-math.pi, math.pi, 360, endpoint=False)


def cast(x, segments, rng):
    """Ranges from (x, 0) heading +x to wall segments [(x1, y1, x2, y2)], 1 cm noise."""
    out = np.full(ANGLES.size, np.inf)
    for i, a in enumerate(ANGLES):
        dx, dy = math.cos(a), math.sin(a)
        for x1, y1, x2, y2 in segments:
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((x1 - x) * ey - y1 * ex) / den
            u = ((x1 - x) * dy - y1 * dx) / den
            if t > 0 and 0 <= u <= 1:
                out[i] = min(out[i], t)
    return out + rng.normal(0, 0.01, out.size)


LANE = [(-3, 0.4, 6, 0.4), (-3, -0.4, 6, -0.4)]
END = LANE + [(2.5, -0.4, 2.5, 0.4)]


def test_measures_travel_toward_a_cross_wall():
    rng = np.random.default_rng(0)
    s = scan_shift((ANGLES, cast(0.0, END, rng)), (ANGLES, cast(0.2, END, rng)))
    assert s is not None and abs(s - 0.2) < 0.03


def test_reverse_travel():
    rng = np.random.default_rng(1)
    s = scan_shift((ANGLES, cast(0.3, END, rng)), (ANGLES, cast(0.15, END, rng)))
    assert s is not None and abs(s + 0.15) < 0.03


def test_stationary_reads_zero():
    rng = np.random.default_rng(2)
    s = scan_shift((ANGLES, cast(0.0, END, rng)), (ANGLES, cast(0.0, END, rng)))
    assert s is not None and abs(s) < 0.02


def test_featureless_corridor_is_unknown():
    rng = np.random.default_rng(3)
    assert scan_shift((ANGLES, cast(0.0, LANE, rng)), (ANGLES, cast(0.2, LANE, rng))) is None
