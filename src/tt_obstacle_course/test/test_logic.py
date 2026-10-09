"""Offline tests for the pure-logic parts (no ROS needed).

A tiny bicycle-model sim with a ray-cast lidar checks behaviors against the
truck's real limitation: a 1.52 m minimum turning radius.
"""

import math

import numpy as np
import pytest

from tt_obstacle_course.behaviors import (DONE, FAILED, RUNNING, Config, Context,
                                          make_behavior)
from tt_obstacle_course.geometry import OdomTracker, SteeringModel, wrap_angle
from tt_obstacle_course.scan_utils import ScanView

R_MIN = 1.52
STEER = SteeringModel(R_MIN, 0.5)
GAINS = {
    'heading_hold': {'kp': 2.0, 'kd': 0.1},
    'wall_center': {'kp': 1.2, 'kd': 0.15, 'lookahead': 0.8},
    'arc_turn': {'yaw_rate_kp': 0.8, 'tolerance_deg': 5.0},
    'k_turn': {'front_clearance': 0.45, 'rear_clearance': 0.40, 'leg_max_distance': 0.6,
               'leg_speed': 0.4, 'heading_tolerance_deg': 10.0, 'max_legs': 30},
}


def raycast(segments, x, y, yaw, n=360, max_range=8.0):
    angles = -math.pi + np.arange(n) * (2 * math.pi / n)
    ranges = np.full(n, np.inf)
    for i, a in enumerate(angles):
        dx, dy = math.cos(yaw + a), math.sin(yaw + a)
        best = max_range + 1
        for (x1, y1), (x2, y2) in segments:
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((x1 - x) * ey - (y1 - y) * ex) / den
            u = ((x1 - x) * dy - (y1 - y) * dx) / den
            if t > 0 and 0 <= u <= 1:
                best = min(best, t)
        if best <= max_range:
            ranges[i] = best
    return ScanView(ranges, -math.pi, 2 * math.pi / n, 0.05, max_range)


class Sim:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, segments=()):
        self.x, self.y, self.yaw = x, y, yaw
        self.segments = list(segments)
        self.odom = OdomTracker(alpha=1.0)
        self.t = 0.0
        self._push()

    def _push(self):
        qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.odom.update(self.t, self.x, self.y, 0.0, 0.0, 0.0, qz, qw)

    def ctx(self):
        scan = raycast(self.segments, self.x, self.y, self.yaw) if self.segments else None
        return Context(t=self.t, odom=self.odom, scan=scan, lidar_ok=scan is not None)

    def step(self, cmd, dt=0.05):
        kappa = STEER.steer_to_curvature(cmd.steer)
        self.x += cmd.speed * math.cos(self.yaw) * dt
        self.y += cmd.speed * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(self.yaw + cmd.speed * kappa * dt)
        self.t += dt
        self._push()

    def run(self, behavior, max_s=60.0, dt=0.05):
        cfg_ctx = self.ctx()
        behavior.start(cfg_ctx)
        while self.t < max_s:
            cmd = behavior.step(self.ctx(), dt)
            if cmd.status != RUNNING:
                return cmd.status
            self.step(cmd, dt)
        return 'timeout'

    def min_wall_distance(self):
        best = math.inf
        for (x1, y1), (x2, y2) in self.segments:
            ex, ey = x2 - x1, y2 - y1
            t = max(0, min(1, ((self.x - x1) * ex + (self.y - y1) * ey) / (ex * ex + ey * ey)))
            best = min(best, math.hypot(self.x - x1 - t * ex, self.y - y1 - t * ey))
        return best


def cfg():
    return Config(STEER, GAINS)


def test_steering_model_round_trip():
    assert STEER.curvature_to_steer(1 / R_MIN) == pytest.approx(0.5)
    assert STEER.curvature_to_steer(10.0) == pytest.approx(0.5)      # clamps at full lock
    assert STEER.steer_to_curvature(-0.25) == pytest.approx(-0.5 / R_MIN)


def test_wall_center_converges_in_corridor():
    # 0.81 m (32 in) corridor along +x, robot starts 0.25 m off centre.
    walls = [((-1, 0.405), (30, 0.405)), ((-1, -0.405), (30, -0.405))]
    sim = Sim(y=0.25, segments=walls)
    beh = make_behavior('wall_center', {'speed': 0.8, 'distance': 8.0}, cfg())
    assert sim.run(beh) == DONE
    assert abs(sim.y) < 0.08
    assert sim.min_wall_distance() > 0.12


def test_arc_turn_helix_270_left():
    sim = Sim()
    beh = make_behavior('arc_turn', {'direction': 'left', 'turn_deg': 270, 'radius': 1.6,
                                     'speed': 0.5, 'allow_kturn': False}, cfg())
    assert sim.run(beh) == DONE
    assert math.degrees(wrap_angle(sim.yaw)) == pytest.approx(-90, abs=8)
    # Centre of the arc is (0, 1.6); stay on the radius.
    assert math.hypot(sim.x, sim.y - 1.6) == pytest.approx(1.6, abs=0.15)


def test_arc_turn_warns_when_radius_below_vehicle_min():
    sim = Sim()
    beh = make_behavior('arc_turn', {'direction': 'left', 'turn_deg': 90, 'radius': 1.2}, cfg())
    beh.start(sim.ctx())
    assert 'min' in beh.warning


def test_k_turn_reverses_in_a_box_too_small_to_u_turn():
    # 2.0 x 2.0 m box: a 180 deg turn needs 3.0 m+ at R=1.52, so it must reverse.
    box = [((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))]
    sim = Sim(segments=box)
    beh = make_behavior('k_turn', {'direction': 'left', 'turn_deg': 180}, cfg())
    assert sim.run(beh, max_s=120) == DONE
    assert abs(wrap_angle(sim.yaw - math.pi)) < math.radians(12)
    assert beh.legs >= 2
    assert sim.min_wall_distance() > 0.15


def test_k_turn_gives_up_after_max_legs():
    box = [((-0.5, -0.5), (0.5, -0.5)), ((0.5, -0.5), (0.5, 0.5)),
           ((0.5, 0.5), (-0.5, 0.5)), ((-0.5, 0.5), (-0.5, -0.5))]
    sim = Sim(segments=box)
    beh = make_behavior('k_turn', {'direction': 'left', 'turn_deg': 180, 'max_legs': 3}, cfg())
    assert sim.run(beh) == FAILED


def test_corridor_clearance_ignores_side_walls():
    # 20 in (0.508 m) lane: an angular cone would see the walls at ~0.44 m.
    lane = [((-2, 0.254), (10, 0.254)), ((-2, -0.254), (10, -0.254))]
    scan = raycast(lane, 0.0, 0.0, 0.0)
    assert scan.front_clearance(math.radians(35)) < 0.5
    assert scan.corridor_clearance(0.18) == math.inf
    blocked = raycast(lane + [((1.0, -0.1), (1.0, 0.1))], 0.0, 0.0, 0.0)
    assert blocked.corridor_clearance(0.18) == pytest.approx(1.0, abs=0.03)
    assert blocked.corridor_clearance(0.18, forward=False) == math.inf


def test_stuck_monitor_backs_off_then_fails():
    from tt_obstacle_course.behaviors import Command, StuckMonitor
    sim = Sim()
    mon = StuckMonitor(min_speed=0.05, time_s=0.5, backoff_distance=0.2, max_retries=1)
    push = Command(0.5, 0.0)
    started = False
    for _ in range(40):                     # commanded forward, never moves
        sim.t += 0.05
        sim._push()
        started |= mon.check(sim.ctx(), push, cfg())
        if mon.active():
            mon.backoff = None              # pretend the backoff finished
    assert started and mon.exhausted()


def test_heading_hold_absolute():
    sim = Sim(yaw=0.3)
    beh = make_behavior('heading_hold', {'heading_abs': 0.0, 'speed': 0.5, 'distance': 4.0}, cfg())
    assert sim.run(beh) == DONE
    assert abs(sim.yaw) < math.radians(3)


def _wall_with(gap=None, notch=None):
    """Left wall at y=1.0 along x, with an opening or a closed notch."""
    segs = []
    x = -1.0
    for kind, (a, b) in sorted([(k, v) for k, v in (('gap', gap), ('notch', notch)) if v],
                               key=lambda kv: kv[1][0]):
        segs.append(((x, 1.0), (a, 1.0)))
        if kind == 'notch':
            segs += [((a, 1.0), (a, 1.4)), ((a, 1.4), (b, 1.4)), ((b, 1.4), (b, 1.0))]
        x = b
    segs.append(((x, 1.0), (8.0, 1.0)))
    segs.append(((-1.0, -1.0), (8.0, -1.0)))
    return segs


def test_side_opening_found_and_decoy_notch_rejected():
    real = raycast(_wall_with(gap=(1.0, 1.7)), 0.0, 0.0, 0.0)
    gaps = real.find_side_openings(+1, min_width=0.45, depth_beyond=1.5)
    assert gaps, 'real opening not detected'
    assert gaps[0]['x'] == pytest.approx(1.35, abs=0.15)
    assert gaps[0]['y'] == pytest.approx(1.0, abs=0.1)
    assert gaps[0]['yaw'] == pytest.approx(math.pi / 2, abs=0.2)

    decoy = raycast(_wall_with(notch=(1.0, 1.7)), 0.0, 0.0, 0.0)
    assert decoy.find_side_openings(+1, min_width=0.45, depth_beyond=1.5) == []


def test_gate_detection_from_two_posts():
    def post(cx, cy, r=0.02):
        return [((cx - r, cy - r), (cx + r, cy - r)), ((cx + r, cy - r), (cx + r, cy + r)),
                ((cx + r, cy + r), (cx - r, cy + r)), ((cx - r, cy + r), (cx - r, cy - r))]
    scan = raycast(post(1.5, 0.3) + post(1.5, -0.3), 0.0, 0.0, 0.0, n=720)
    gates = scan.find_gates(gate_width=0.6, tol=0.1)
    assert gates
    assert gates[0]['x'] == pytest.approx(1.5, abs=0.05)
    assert gates[0]['y'] == pytest.approx(0.0, abs=0.05)
    assert gates[0]['yaw'] == pytest.approx(0.0, abs=0.1)
