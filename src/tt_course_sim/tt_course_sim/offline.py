"""Fast closed-loop testing of one behavior on one stretch of the course, no ROS.

Uses the same course model, lidar ray-marching, rear-axle kinematics and the
real behaviors + governor collision strip, so a section can be tuned in
seconds instead of a full ROS run.

  python3 -m tt_course_sim.offline narrow           # run a named scenario
  python3 -m tt_course_sim.offline all              # run every scenario
  python3 -m tt_course_sim.offline all --radius=1.2 # what-if turning radius
"""

import math
import sys

import yaml

from tt_course_sim.course import FALL_DROP, Course
from tt_obstacle_course.behaviors import RUNNING, Config, Context, StuckMonitor, make_behavior
from tt_obstacle_course.geometry import OdomTracker, SteeringModel
from tt_obstacle_course.scan_utils import ScanView


def load_tuning(path='src/tt_obstacle_course/config/tuning.yaml'):
    with open(path, encoding='utf-8') as f:
        t = yaml.safe_load(f)
    shared = t['/**']['ros__parameters']
    ex = t['behavior_executor']['ros__parameters']
    gov = t['speed_governor']['ros__parameters']
    return shared, ex, gov


class Truck:

    def __init__(self, course, x, y, yaw, radius=1.52, z=0.0, rear_axle=0.38):
        self.c = course
        self.x, self.y, self.yaw, self.z = x, y, yaw, z
        self.v = self.steer = 0.0
        self.radius = radius
        self.rear_axle = rear_axle
        self.collisions = 0
        self.fell = False
        self.t = 0.0
        self.odom = OdomTracker(alpha=0.4)
        self.nose_up = self.roll = 0.0
        self._push()

    def _push(self):
        qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.odom.update(self.t, self.x, self.y, self.z, 0, 0, qz, qw)

    def scan(self):
        lx, ly = self.x - 0.15 * math.cos(self.yaw), self.y - 0.15 * math.sin(self.yaw)
        r = self.c.scan(lx, ly, self.z + 0.25, self.yaw, self.nose_up, self.roll)
        return ScanView(r, -math.pi, 2 * math.pi / r.size, 0.15, 8.0)

    def step(self, speed, steer, dt=0.01):
        self.steer += (steer - self.steer) * min(1.0, dt / 0.08)
        self.v += (speed - self.v) * min(1.0, dt / 0.25)
        kappa = max(-1, min(1, self.steer / 0.5)) / self.radius
        ra = self.rear_axle
        rx, ry = self.x - ra * math.cos(self.yaw), self.y - ra * math.sin(self.yaw)
        ds = self.v * dt
        nyaw = self.yaw + ds * kappa
        rx += ds * math.cos((self.yaw + nyaw) / 2)
        ry += ds * math.sin((self.yaw + nyaw) / 2)
        nx, ny = rx + ra * math.cos(nyaw), ry + ra * math.sin(nyaw)
        z, gx, gy, _, _ = self.c.ground(nx, ny, self.z)
        c, s = math.cos(nyaw), math.sin(nyaw)
        us = [0.0, 0.0, 0.0, -0.22, -0.43, -0.43, -0.43, -0.22]
        vs = [0.1015, 0.0, -0.1015, -0.1015, -0.1015, 0.0, 0.1015, 0.1015]
        fx = [nx + c * u - s * v for u, v in zip(us, vs)]
        fy = [ny + s * u + c * v for u, v in zip(us, vs)]
        self.t += dt
        if z - self.z > 0.06 or self.c.blocked(fx, fy, z):
            self.collisions += 1
            self.v = 0.0
            self._push()
            return False
        if self.z - z > FALL_DROP:
            self.fell = True
        self.x, self.y, self.yaw, self.z = nx, ny, math.atan2(s, c), z
        h = (c, s)
        self.nose_up = math.atan(gx * h[0] + gy * h[1])
        self.roll = math.atan(-gx * h[1] + gy * h[0])
        self._push()
        return True


def run(course, pose, behavior, params, max_s=40.0, radius=None, z=0.0, verbose=False, truck=None,
        stop=None):
    shared, ex, gov = load_tuning()
    veh = dict(shared['vehicle'])
    if radius is None:
        radius = veh['planning_min_turn_radius']
    veh['planning_min_turn_radius'] = radius
    cfg = Config(SteeringModel(veh['planning_min_turn_radius'], veh['max_steer_cmd']),
                 {k: ex[k] for k in ('heading_hold', 'wall_center', 'gap_follow', 'arc_turn',
                                     'k_turn')},
                 half_width=veh['width'] / 2, sweep_margin=ex.get('sweep_margin', 0.15),
                 front_overhang=veh['lidar_to_front'], rear_overhang=veh['lidar_to_rear'],
                 odom_to_front=max(0.0, veh['lidar_to_front'] - 0.15),
                 odom_to_rear=veh['lidar_to_rear'] + 0.15,
                 axle_back=veh['lidar_ahead_of_axle'])
    truck = truck or Truck(course, *pose, radius=radius, z=z)
    t_end = truck.t + max_s
    beh = make_behavior(behavior, params, cfg)
    stuck = StuckMonitor(**{k: ex['stuck'][k] for k in ('min_speed', 'time_s',
                                                         'backoff_distance', 'max_retries')})
    corridor = veh['width'] / 2 + gov['corridor_margin']
    scan = truck.scan()
    beh.start(Context(t=0.0, odom=truck.odom, scan=scan))
    status, last_scan, collisions_logged = RUNNING, -1.0, 0
    cmd = None
    out_steer = 0.0
    steer_rate = gov['steer_rate_limit']
    while truck.t < t_end and not truck.fell:
        if truck.t - last_scan >= 0.1:
            scan, last_scan = truck.scan(), truck.t
        if int(truck.t * 100) % 5 == 0 or cmd is None:
            lidar_ok = max(abs(truck.nose_up), abs(truck.roll)) < math.radians(4)
            if stop is not None and stop(truck, scan, lidar_ok):
                status = 'stopped'
                break
            ctx = Context(t=truck.t, odom=truck.odom, scan=scan, lidar_ok=lidar_ok)
            if stuck.active():
                cmd = stuck.step(ctx, 0.05)
            else:
                cmd = beh.step(ctx, 0.05)
                if cmd.status != RUNNING:
                    status = cmd.status
                    break
                if not getattr(beh, 'self_recovers', False) and stuck.check(ctx, cmd, cfg):
                    cmd = stuck.step(ctx, 0.05)
                elif stuck.exhausted():
                    status = 'failed(stuck)'
                    break
            v = cmd.speed
            k = cfg.steering.steer_to_curvature(out_steer)
            if lidar_ok and v > 0:
                d = scan.corridor_clearance(corridor, True, min_x=veh['lidar_to_front'],
                                            curvature=k, axle_back=cfg.axle_back,
                                            body_half=veh['width'] / 2)
                if d <= gov['stop_distance']:
                    v = 0.0
                elif d < gov['slow_distance']:
                    v = min(v, max(gov['crawl_speed'], (d - gov['stop_distance']) / gov['ttc_brake_s']))
            elif lidar_ok and v < 0:
                d = scan.corridor_clearance(corridor, False, min_x=veh['lidar_to_rear'],
                                            curvature=k, axle_back=cfg.axle_back,
                                            body_half=veh['width'] / 2)
                if d <= gov['rear_stop_distance']:
                    v = 0.0
        out_steer += max(-steer_rate * 0.01, min(steer_rate * 0.01, cmd.steer - out_steer))
        truck.step(v, out_steer)
        if truck.collisions > collisions_logged and verbose:
            print(f'  collision t={truck.t:.1f} at ({truck.x:.2f}, {truck.y:.2f})')
        collisions_logged = truck.collisions
    if status == RUNNING:
        status = 'fell' if truck.fell else 'timeout'
    return status, truck


SCENARIOS = {
    # name: (pose function, behavior, params, success check) or a chained function
    'tunnel': (lambda c: c.waypoint('tunnel_mid'), 'wall_center', {'speed': 1.0, 'distance': 2.5},
               'done'),
    'narrow': (lambda c: c.waypoint('narrow'), 'wall_center', {'speed': 0.8, 'distance': 9.2},
               'done'),
    'gravel': (lambda c: c.waypoint('gravel'), 'wall_center', {'speed': 0.8, 'distance': 5.5},
               'done'),
    'wide_turn': (lambda c: c.waypoint('wide'), 'arc_turn',
                  {'direction': 'left', 'turn_deg': 90, 'radius': 1.6, 'speed': 0.6,
                   'allow_kturn': True}, 'done'),
    # Same parameters as the buckets fallback in mission.yaml.
    'buckets': (lambda c: c.waypoint('buckets'), 'gap_follow',
                {'speed': 0.4, 'goal_heading_deg': 0, 'goal_heading_gain': 1.0, 'max_range': 1.5,
                 'free_range': 0.5, 'bubble_radius': 0.25, 'min_width': 0.3, 'distance': 6.0},
                'done'),
}


def bank_chain(course, radius):
    """mission.yaml bank_approach + bank_uturn: walls_end trigger, fence the plane from its
    known size (as MissionManager.set_bank_frame), then plan_drive into the return lane."""
    with open('src/tt_obstacle_course/config/mission.yaml', encoding='utf-8') as f:
        b = yaml.safe_load(f).get('bank_entry', {})
    r = radius or load_tuning()[0]['vehicle']['planning_min_turn_radius']
    seen = []

    def walls_end(t, s, ok):
        e = s.walls_end() if ok and t.odom.distance >= 0.8 else None
        if e is not None and e <= b['trigger']:
            seen.append(e)
            return True
        return False
    status, truck = run(course, course.waypoint('bank_approach'), 'wall_center', {'speed': 0.6},
                        radius=r, stop=walls_end)
    if status != 'stopped':
        return 'no_trigger', truck
    o = truck.odom
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    w = lambda u, v: [o.x + c * u - s * v, o.y + s * u + c * v]  # noqa: E731
    near = seen[0] - b['lidar_behind_camera']
    far = near + b['plane_depth'] - b['margin']
    fence = [w(far, b['north']) + w(far, b['south']), w(near, b['north']) + w(far, b['north']),
             w(near, b['south']) + w(far, b['south'])]
    goal = w(near - 0.6, b['return_offset']) + [o.yaw + math.pi]
    xs = []
    status, truck = run(course, None, 'plan_drive',
                        {'goal': goal, 'fence': fence, 'speed': 0.45, 'reverse_speed': 0.3,
                         'goal_tol': 0.35, 'yaw_tol': 0.6, 'margin': 1.2},
                        radius=r, truck=truck, max_s=60.0,
                        stop=lambda t, s, ok: xs.append(t.x) and False)
    bx, _, _, hl, _ = course.bank_box
    if max(xs) + 0.05 > bx + hl:
        return 'off_plane', truck
    return status, truck


CHAINS = {'bank': bank_chain}


def wide_gap_chain(course, radius):
    """mission.yaml wide zone: left turn, follow the left wall $turn_in_offset out until a
    left opening >= min_from_start beyond the search start (as MissionManager.fresh_gap),
    then plan_drive into it along the wall normal (as set_entrance + entry: plan)."""
    shared = load_tuning()[0]
    r = radius or shared['vehicle']['planning_min_turn_radius']
    status, truck = run(course, course.waypoint('wide')[:3], 'arc_turn',
                        {'direction': 'left', 'turn_deg': 90, 'radius': 1.6, 'speed': 0.6,
                         'allow_kturn': True}, radius=r)
    if status != 'done':
        return 'turn_' + status, truck
    sx, sy, syaw = truck.x, truck.y, truck.yaw
    min_gap = shared['vehicle']['width'] + 0.25
    found, track = [], []

    def gap_ahead(t, s, ok):
        if not ok:
            return False
        lx, ly = t.x - 0.15 * math.cos(t.yaw), t.y - 0.15 * math.sin(t.yaw)
        # As course_perception: first opening within the width/range caps, 3 matching sightings.
        dets = [g for g in s.find_side_openings(+1, min_gap, 1.5, a_to=math.radians(110))
                if g['width'] <= 1.4 and math.hypot(g['x'], g['y']) <= 3.5]
        if not dets:
            track.clear()
            return False
        g = dets[0]
        gx = lx + g['x'] * math.cos(t.yaw) - g['y'] * math.sin(t.yaw)
        gy = ly + g['x'] * math.sin(t.yaw) + g['y'] * math.cos(t.yaw)
        if track and math.hypot(gx - track[-1][0], gy - track[-1][1]) > 0.3:
            track.clear()
        track.append((gx, gy))
        if len(track) < 3:
            return False
        ahead = (gx - t.x) * math.cos(t.yaw) + (gy - t.y) * math.sin(t.yaw)
        from_start = (gx - sx) * math.cos(syaw) + (gy - sy) * math.sin(syaw)
        if ahead > 0.3 and from_start >= 1.0:
            found.append((gx, gy))
            return True
        return False
    offset = r + 0.38 + 0.3
    status, truck = run(course, None, 'wall_center',
                        {'speed': 0.5, 'follow': 'left', 'wall_target': offset, 'max_wall': 2.6,
                         'distance': 8.0}, radius=r, truck=truck, stop=gap_ahead)
    if not found:
        return 'no_gap', truck
    gx, gy = truck.gap_seen = found[0]
    axis = round((truck.yaw + math.pi / 2) / (math.pi / 2)) * (math.pi / 2)
    goal = [gx + 0.6 * math.cos(axis), gy + 0.6 * math.sin(axis), axis]
    return run(course, None, 'plan_drive', {'goal': goal, 'speed': 0.5}, radius=r, truck=truck,
               max_s=60.0)


CHAINS = {'bank': bank_chain}


def _in_return_lane(course, truck):
    # Heading back west inside the 32" return lane at the pothole ramp (within the
    # plan's yaw tolerance; the pothole lane follower straightens the rest).
    x0, y0, _ = course.waypoint('bank_approach')
    return abs(truck.y - (y0 - 1.495)) < 0.45 and math.cos(truck.yaw) < -0.7 and truck.x < x0 + 3.0


def _through_narrow(course, truck):
    gx, gy, gyaw = course.waypoint('gravel')
    dx, dy = truck.x - gx, truck.y - gy
    return dx * math.cos(gyaw) + dy * math.sin(gyaw) > -0.5 and math.cos(truck.yaw - gyaw) > 0.94


def _through_buckets(course, truck):
    hx, hy, hyaw = course.waypoint('hoops')
    dx, dy = truck.x - hx, truck.y - hy
    return dx * math.cos(hyaw) + dy * math.sin(hyaw) > -0.2


ENDS = {'bank': _in_return_lane, 'narrow': _through_narrow, 'buckets': _through_buckets}


def _in_bucket_field(course, truck):
    # Through the entrance: east of the shared wall, within the field's north-south span.
    return truck.x > -2.96 + 0.2 and 0.85 < truck.y < 0.85 + 4.13


def sweep_entrances(radius, positions=(0.46, 0.8, 1.3, 1.9, 2.5, 3.1, 3.67)):
    """Entrance anywhere along the wall shared by the wide area and the bucket field."""
    ok = True
    for b in positions:
        course = Course(seed=1, gap_b=b)
        status, truck = wide_gap_chain(course, radius)
        seen = getattr(truck, 'gap_seen', (float('nan'),) * 2)
        good = (status == 'done' and truck.collisions == 0 and _in_bucket_field(course, truck)
                 and abs(seen[1] - (0.85 + b)) < 0.35)
        ok &= good
        print(f'{"PASS" if good else "FAIL"} wide_gap b={b:4.2f} status={status:10s} t={truck.t:5.1f}s '
              f'collisions={truck.collisions} gap_err={seen[1] - 0.85 - b:+.2f} end=({truck.x:.2f}, {truck.y:.2f}, '
              f'{math.degrees(truck.yaw):.0f} deg)')
    return ok


def main(argv):
    radius = None
    for a in list(argv):
        if a.startswith('--radius='):
            radius = float(a.split('=', 1)[1])
            argv.remove(a)
    names = list(SCENARIOS) + list(CHAINS) if not argv or argv[0] == 'all' else argv
    if names == ['entrances']:
        return 0 if sweep_entrances(radius) else 1
    course = Course()
    ok = True
    for name in names:
        if name in CHAINS:
            status, truck = CHAINS[name](course, radius)
            want = 'done'
        else:
            pose_fn, beh, params, want = SCENARIOS[name]
            pose = pose_fn(course)
            status, truck = run(course, pose[:3], beh, params, z=pose[3] if len(pose) > 3 else 0.0,
                                radius=radius, verbose=True)
        good = status == want and truck.collisions == 0
        check = ENDS.get(name)
        if check:
            good &= check(course, truck)
        ok &= good
        print(f'{"PASS" if good else "FAIL"} {name:14s} status={status:14s} t={truck.t:5.1f}s '
              f'collisions={truck.collisions} end=({truck.x:.2f}, {truck.y:.2f}, '
              f'{math.degrees(truck.yaw):.0f} deg)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
