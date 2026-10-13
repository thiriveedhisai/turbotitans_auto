"""Start-light disc detection and the lidar self-mask (no ROS needed)."""
import math

import cv2
import numpy as np
import pytest

from tt_obstacle_course.light_blob import RED_RANGES, bright_exit, colour_mask, disc_centres
from tt_obstacle_course.scan_utils import ScanView

GREEN = [([40, 80, 60], [85, 255, 255])]
BGR_GREEN, BGR_RED, BGR_STICK = (40, 200, 40), (30, 30, 220), (90, 140, 170)


def _found(draw, ranges=GREEN):
    img = np.full((480, 640, 3), 200, np.uint8)
    draw(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return disc_centres(colour_mask(hsv, ranges), min_area=400)


def test_round_disc():
    c = _found(lambda im: cv2.circle(im, (320, 200), 60, BGR_GREEN, -1))
    assert c and abs(c[0][0] - 320) < 3 and abs(c[0][1] - 200) < 3


def test_slightly_oval_disc():
    assert _found(lambda im: cv2.ellipse(im, (320, 200), (70, 50), 15, 0, 360, BGR_GREEN, -1))


def test_disc_on_a_wooden_stick():
    def draw(im):
        cv2.rectangle(im, (312, 240), (328, 470), BGR_STICK, -1)
        cv2.circle(im, (320, 200), 60, BGR_GREEN, -1)
    assert _found(draw)


def test_disc_on_a_same_colour_stick_centred_on_the_disc():
    def draw(im):
        cv2.rectangle(im, (310, 240), (330, 470), BGR_GREEN, -1)
        cv2.circle(im, (320, 200), 60, BGR_GREEN, -1)
    c = _found(draw)
    assert c and abs(c[0][1] - 200) < 12


def test_long_bar_is_not_a_disc():
    assert not _found(lambda im: cv2.rectangle(im, (100, 190), (540, 215), BGR_GREEN, -1))


def test_red_disc_seen_as_red_not_green():
    draw = lambda im: cv2.circle(im, (320, 200), 60, BGR_RED, -1)  # noqa: E731
    assert _found(draw, RED_RANGES)
    assert not _found(draw)


def test_bright_exit_bearing_in_tunnel():
    # Dark tunnel (walls ~40, textured), exit 60 px left of centre and much brighter.
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(40, 8, (360, 640)), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (230, 185), (290, 250), 220, -1)
    fx, cx = 320.0, 320.0
    b = bright_exit(img, fx, cx)
    assert b is not None
    assert b == pytest.approx(math.atan2(60, fx), abs=math.radians(3))


def test_bright_exit_none_in_the_open():
    # Evenly lit course: a sunny patch isn't 1.8x the scene.
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(150, 20, (360, 640)), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (400, 200), (460, 260), 200, -1)
    assert bright_exit(img, 320.0, 320.0) is None


def test_self_mask_drops_only_the_flag():
    # Flag pole 0.178 m behind, 0.10 m right of the lidar; a real wall 1 m to the right
    # and a cone 0.6 m out in the pole's direction (beyond max_range: kept).
    n = 720
    a = -math.pi + np.arange(n) * 2 * math.pi / n
    r = np.full(n, np.inf)
    bearing = math.atan2(-0.10, -0.178)
    flag = np.argmin(np.abs(a - bearing))
    r[flag] = math.hypot(0.178, 0.10)
    r[flag + 2] = 0.6
    wall = np.abs(a + math.pi / 2) < math.radians(10)
    r[wall] = 1.0 / np.cos(a[wall] + math.pi / 2)
    masked = ScanView(r, -math.pi, 2 * math.pi / n, 0.05, 12.0,
                      self_mask=[(math.degrees(bearing), 8.0, 0.28)])
    assert np.isfinite(masked.ranges).sum() == wall.sum() + 1
    assert np.isclose(masked.ranges, 0.6).sum() == 1
    assert abs(masked.side_distance(-1) - 1.0) < 0.01
