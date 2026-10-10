"""Depth-image hoop check (tt_obstacle_course.depth_check) on synthetic depth renders."""

import math

import numpy as np

from tt_obstacle_course.depth_check import DepthView, leg_points

K = [[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]]
CAM_H = 0.20


def render(objects, background=6.0, w=640, h=360):
    """Depth image of vertical/horizontal boxes in front of a flat backdrop.

    objects: (x, y0, y1, z0, z1) -> a surface at forward distance x spanning lateral
    y0..y1 (left +) and height z0..z1 above ground."""
    d = np.full((h, w), background, np.float32)
    v, u = np.mgrid[0:h, 0:w]
    for x, y0, y1, z0, z1 in sorted(objects, key=lambda o: -o[0]):
        y = -(u - K[0][2]) * x / K[0][0]
        z = CAM_H - (v - K[1][2]) * x / K[1][1]
        d[(y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)] = x
    return d


def hoop(x, yc, width=0.64, leg=0.6, r=0.025):
    half = width / 2
    parts = [(x, yc + half - r, yc + half + r, 0.0, leg), (x, yc - half - r, yc - half + r, 0.0, leg)]
    for k in range(12):
        a = math.pi * (k + 0.5) / 12
        yy, zz = yc + half * math.cos(a), leg + half * math.sin(a)
        parts.append((x, yy - 0.04, yy + 0.04, zz - 0.03, zz + 0.03))
    return parts


def gate_at(x, yc, width=0.64):
    return {'x': x, 'y': yc, 'yaw': 0.0, 'width': width}


def legs_of(g):
    return list(leg_points(g))


def test_leg_points_straddle_the_gate():
    (x1, y1), (x2, y2) = leg_points(gate_at(2.0, 0.1, 0.6))
    assert abs(x1 - 2.0) < 1e-9 and abs(x2 - 2.0) < 1e-9
    assert abs(abs(y1 - y2) - 0.6) < 1e-9 and abs((y1 + y2) / 2 - 0.1) < 1e-9


def test_hoop_is_confirmed():
    view = DepthView(render(hoop(2.0, 0.1)), K, CAM_H)
    assert view.hoop_frame(legs_of(gate_at(2.0, 0.1))) is True


def test_wall_end_pair_is_rejected():
    # Two thin uprights with a solid bale wall behind/between them: no see-through opening.
    objs = [(2.0, 0.39, 0.45, 0.0, 0.6), (2.0, -0.25, -0.19, 0.0, 0.6), (2.1, -1.0, 1.0, 0.0, 0.45)]
    view = DepthView(render(objs), K, CAM_H)
    assert view.hoop_frame(legs_of(gate_at(2.0, 0.1))) is False


def test_two_posts_without_top_are_rejected():
    objs = [(2.0, 0.39, 0.45, 0.0, 0.6), (2.0, -0.25, -0.19, 0.0, 0.6)]
    view = DepthView(render(objs), K, CAM_H)
    assert view.hoop_frame(legs_of(gate_at(2.0, 0.1))) is False


def test_too_close_to_see_the_top_is_undecided():
    for x in (0.5, 0.8):
        view = DepthView(render(hoop(x, 0.0)), K, CAM_H)
        assert view.hoop_frame(legs_of(gate_at(x, 0.0))) is None


def test_confirmed_from_1_to_3_5_m_for_20_and_30_inch_hoops():
    for x in (1.2, 2.0, 3.0, 3.5):
        for width in (0.51, 0.76):
            view = DepthView(render(hoop(x, 0.1, width=width)), K, CAM_H)
            assert view.hoop_frame(legs_of(gate_at(x, 0.1, width))) is True
