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
    'wall_center': {'lookahead': 0.0},
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
    # Sim rotates about the sensor point, so no rear-axle offset.
    return Config(STEER, GAINS, axle_back=0.0)


def arc_walls(cx, cy, r, a0, a1, n=40):
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n))
           for k in range(n + 1)]
    return list(zip(pts[:-1], pts[1:]))


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


def test_wall_center_narrow_lane_fast_offset_start():
    # 20 in lane, 0.1 m off centre and 10 deg off axis at 1.0 m/s: must settle, not weave.
    walls = [((-1, 0.254), (30, 0.254)), ((-1, -0.254), (30, -0.254))]
    sim = Sim(y=0.1, yaw=math.radians(-10), segments=walls)
    beh = make_behavior('wall_center', {'speed': 1.0, 'distance': 6.0}, cfg())
    worst = math.inf
    beh.start(sim.ctx())
    while sim.t < 20:
        cmd = beh.step(sim.ctx(), 0.05)
        if cmd.status != RUNNING:
            break
        sim.step(cmd)
        worst = min(worst, sim.min_wall_distance())
    assert cmd.status == DONE
    assert abs(sim.y) < 0.03
    assert worst > 0.12          # half the truck width + margin


def test_wall_center_keeps_lane_width_when_a_wall_ends():
    # Left wall stops at x=2 (opening): keep centred using the remembered width.
    walls = [((-1, 0.405), (2, 0.405)), ((-1, -0.405), (30, -0.405))]
    sim = Sim(y=0.0, segments=walls)
    beh = make_behavior('wall_center', {'speed': 0.8, 'distance': 6.0}, cfg())
    assert sim.run(beh) == DONE
    assert abs(sim.y) < 0.06


def test_wall_center_follows_a_curved_lane():
    # 28 in lane bending 90 deg right on a 6 ft centreline radius (the narrow curve).
    r, w = 1.83, 0.71
    walls = ([((-1, w / 2), (0, w / 2)), ((-1, -w / 2), (0, -w / 2))]
             + arc_walls(0, -r, r + w / 2, math.pi / 2, 0) + arc_walls(0, -r, r - w / 2, math.pi / 2, 0)
             + [((r + w / 2, -r), (r + w / 2, -r - 3)), ((r - w / 2, -r), (r - w / 2, -r - 3))])
    sim = Sim(x=-0.8, segments=walls)
    beh = make_behavior('wall_center', {'speed': 0.6, 'distance': 5.5}, cfg())
    worst = math.inf
    beh.start(sim.ctx())
    while sim.t < 20:
        cmd = beh.step(sim.ctx(), 0.05)
        if cmd.status != RUNNING:
            break
        sim.step(cmd)
        worst = min(worst, sim.min_wall_distance())
    assert cmd.status == DONE
    assert sim.y < -r - 0.3 and abs(sim.x - r) < 0.12
    assert worst > 0.12


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


def test_corridor_clearance_margin_tapers_from_bumper():
    # Lane wall 0.15 m to the side (5 cm off an 8" truck), bumper 0.152 m ahead of the
    # lidar: a full 0.18 m strip read it as an obstacle 0.155 m ahead and stopped the
    # truck in every tight lane. The margin grows from the body over 0.15 m; a real box
    # ahead in the lane is still seen.
    wall = [((-2, 0.15), (10, 0.15))]
    scan = raycast(wall, 0.0, 0.0, 0.0)
    assert scan.corridor_clearance(0.18, min_x=0.152) < 0.17
    # Tapered: only where the full margin is reached (~0.25 m), beyond stop-at-bumper.
    assert scan.corridor_clearance(0.18, min_x=0.152, body_half=0.1015) > 0.24
    box = raycast([((0.6, -0.05), (0.6, 0.05))], 0.0, 0.0, 0.0)
    assert box.corridor_clearance(0.18, min_x=0.152, body_half=0.1015) == pytest.approx(0.6, abs=0.03)


def test_corridor_clearance_arc_follows_steering():
    # Wall 0.16 m to the left, parallel (3-D sim tunnel stop): a straight strip
    # says blocked; steering right away from it is clear, steering left into it is not.
    wall = [((-2, 0.16), (10, 0.16))]
    scan = raycast(wall, 0.0, 0.0, 0.0)
    assert scan.corridor_clearance(0.18, min_x=0.2) < 0.5
    k = 1 / 1.2
    assert scan.corridor_clearance(0.18, min_x=0.2, curvature=-k, axle_back=0.23) == math.inf
    assert scan.corridor_clearance(0.18, min_x=0.2, curvature=k, axle_back=0.23) < 0.8
    # Box straight ahead at 1 m is still seen at gentle curvature, at ~ arc length 1 m.
    box = raycast([((1.0, -0.15), (1.0, 0.15))], 0.0, 0.0, 0.0)
    assert box.corridor_clearance(0.18, curvature=0.2, axle_back=0.23) == pytest.approx(1.0, abs=0.1)
    # Reverse with left lock: the rear axle runs back along the same circle, curving
    # left. A box behind-left, outside the straight strip, is in the way.
    rear = raycast([((-0.9, 0.25), (-0.5, 0.25))], 0.0, 0.0, 0.0)
    assert rear.corridor_clearance(0.18, forward=False, min_x=0.3) == math.inf
    assert rear.corridor_clearance(0.18, forward=False, min_x=0.3, curvature=k,
                                   axle_back=0.23) < 1.0


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


def test_wall_break_found_when_hugging_wall():
    # Traced 0.5 m off the wall: a 36" door 1.5 m ahead, with a bucket 0.8 m inside it
    # (the edge-run method ended on the bucket and read a 3 m opening).
    segs = [((-1.0, 0.5), (1.05, 0.5)), ((1.96, 0.5), (8.0, 0.5)),
            ((1.3, 1.3), (1.6, 1.3)), ((-1.0, -2.8), (8.0, -2.8))]
    gaps = raycast(segs, 0.0, 0.0, 0.0, n=720).find_wall_breaks(+1, 0.45, 1.0)
    assert gaps, 'door not detected'
    assert gaps[0]['x'] == pytest.approx(1.5, abs=0.08)
    assert gaps[0]['width'] == pytest.approx(0.91, abs=0.12)
    assert gaps[0]['yaw'] == pytest.approx(math.pi / 2, abs=0.1)


def test_wall_break_rejects_notch_and_shadow():
    # Closed 0.65 m notch (zig-zag decoy depth), and a post standing in front of the wall:
    # neither has beams passing 1 m beyond the wall line.
    notch = [((-1.0, 0.5), (1.0, 0.5)), ((1.0, 0.5), (1.0, 1.15)), ((1.0, 1.15), (1.9, 1.15)),
             ((1.9, 1.15), (1.9, 0.5)), ((1.9, 0.5), (8.0, 0.5))]
    assert raycast(notch, 0.0, 0.0, 0.0, n=720).find_wall_breaks(+1, 0.45, 1.0) == []
    post = [((-1.0, 0.5), (8.0, 0.5)), ((0.9, 0.3), (1.2, 0.3)), ((0.9, 0.3), (0.9, 0.5))]
    assert raycast(post, 0.0, 0.0, 0.0, n=720).find_wall_breaks(+1, 0.45, 1.0) == []


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


def _post(cx, cy, r=0.02):
    return [((cx - r, cy - r), (cx + r, cy - r)), ((cx + r, cy - r), (cx + r, cy + r)),
            ((cx + r, cy + r), (cx - r, cy + r)), ((cx - r, cy + r), (cx - r, cy - r))]


def test_gate_with_outer_leg_close_to_wall():
    # Hoop 3: outer leg ~0.17 m inside the U-turn wall, which runs past it.
    wall = [((0.5, 0.49), (3.0, 0.49))]
    scan = raycast(_post(1.5, 0.3) + _post(1.5, -0.3) + wall, 0.0, 0.0, 0.0, n=720)
    gates = scan.find_gates(gate_width=0.6, tol=0.1, isolation=0.12)
    assert gates
    assert gates[0]['y'] == pytest.approx(0.0, abs=0.05)


def test_gate_with_leg_merged_into_thick_wall():
    # Gazebo geometry: 0.12 m bale wall whose face is 0.05 m beyond the leg's edge,
    # closer than cluster_gap, so the leg only shows as a step out of the wall.
    wall = [((-1.0, 0.375), (3.0, 0.375)), ((-1.0, 0.495), (3.0, 0.495))]
    scan = raycast(_post(1.5, 0.3, 0.025) + _post(1.5, -0.33, 0.025) + wall, 0.0, 0.0, 0.0, n=360)
    gates = scan.find_gates(gate_width=0.635, tol=0.18, isolation=0.12)
    assert gates
    assert gates[0]['y'] == pytest.approx(-0.015, abs=0.06)


def test_gate_not_formed_from_doorway_in_wall():
    # Two wall ends 0.6 m apart (a doorway) are not hoop legs.
    walls = [((1.5, 0.3), (1.5, 2.0)), ((1.5, -0.3), (1.5, -2.0))]
    scan = raycast(walls, 0.0, 0.0, 0.0, n=720)
    assert scan.find_gates(gate_width=0.6, tol=0.1, isolation=0.12) == []


def _box_map(segments):
    from tt_obstacle_course.planner import ObstacleMap
    m = ObstacleMap()
    for (x1, y1), (x2, y2) in segments:
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 0.03))
        for t in np.linspace(0, 1, n):
            m.cells.add((int(math.floor((x1 + t * (x2 - x1)) / m.res)),
                         int(math.floor((y1 + t * (y2 - y1)) / m.res))))
    return m


def test_blocked_goal_nudged_into_connector():
    from tt_obstacle_course.planner import Footprint, free_goal
    # 0.91 m connector along x between walls at y=0 and y=0.91; goal seen 0.25 m off-centre.
    m = _box_map([((0, 0), (3, 0)), ((0, 0.91), (3, 0.91))])
    fp = Footprint(rear=0.1, front=0.45, half_width=0.13, margin=0.05)
    grid = m.inflated(-1, -1, 4, 2, fp.radius)
    goal = (1.5, 0.2, 0.0)
    assert free_goal((1.5, 0.455, 0.0), grid, fp) == (1.5, 0.455, 0.0)
    nudged = free_goal(goal, grid, fp)
    assert nudged is not None and 0.3 <= nudged[1] <= 0.61


def test_planner_threads_a_narrow_exit_and_respects_turning_radius():
    from tt_obstacle_course.planner import Footprint, path_blocked, plan
    # 3.6 x 4 m field, 36" exit in the east wall off the start axis, a bucket in the way.
    walls = [((0, 0), (3.6, 0)), ((0, 4), (3.6, 4)), ((0, 0), (0, 4)),
             ((3.6, 0), (3.6, 1.0)), ((3.6, 1.91), (3.6, 4))]
    bucket = [((1.8 + 0.15 * math.cos(a), 2.0 + 0.15 * math.sin(a)),
               (1.8 + 0.15 * math.cos(a + 0.5), 2.0 + 0.15 * math.sin(a + 0.5)))
              for a in np.arange(0, 2 * math.pi, 0.5)]
    m = _box_map(walls + bucket)
    fp = Footprint()
    grid = m.inflated(-0.5, -0.5, 5.0, 4.5, fp.radius)
    path = plan((0.4, 3.0, 0.0), (3.9, 1.455, 0.0), grid, fp, 1.07)
    assert path is not None
    assert not path_blocked(path, grid, fp)
    # Every step is a straight or a full-lock arc: |dyaw| <= arc length / R (arc >= chord).
    for (x0, y0, a0, _), (x1, y1, a1, _) in zip(path, path[1:]):
        ds = math.hypot(x1 - x0, y1 - y0)
        assert abs(wrap_angle(a1 - a0)) <= 1.01 * ds / 1.07 + 1e-6


def test_planner_rejects_goal_inside_obstacle():
    from tt_obstacle_course.planner import Footprint, plan
    m = _box_map([((2.0, -1.0), (2.0, 1.0))])
    fp = Footprint()
    grid = m.inflated(-1.0, -2.0, 4.0, 2.0, fp.radius)
    assert plan((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), grid, fp, 1.07) is None
