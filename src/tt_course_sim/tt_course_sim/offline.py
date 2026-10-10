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
from tt_obstacle_course.geometry import OdomTracker, SteeringModel, bank_lead
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
        us = [0.05, 0.05, 0.05, -0.2, -0.45, -0.45, -0.45, -0.2]
        vs = [0.1, 0.0, -0.1, -0.1, -0.1, 0.0, 0.1, 0.1]
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
                                            curvature=k, axle_back=cfg.axle_back)
                if d <= gov['stop_distance']:
                    v = 0.0
                elif d < gov['slow_distance']:
                    v = min(v, max(gov['crawl_speed'], (d - gov['stop_distance']) / gov['ttc_brake_s']))
            elif lidar_ok and v < 0:
                d = scan.corridor_clearance(corridor, False, min_x=veh['lidar_to_rear'],
                                            curvature=k, axle_back=cfg.axle_back)
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
    """mission.yaml bank_approach + bank_uturn: walls_end trigger, $bank_lead, semicircle."""
    with open('src/tt_obstacle_course/config/mission.yaml', encoding='utf-8') as f:
        mission = yaml.safe_load(f)
    entry = mission.get('bank_entry', {})
    r = radius or load_tuning()[0]['vehicle']['planning_min_turn_radius']
    trig = float(entry.get('trigger', 1.2))

    def walls_end(t, s, ok):
        e = s.walls_end() if ok and t.odom.distance >= 0.8 else None
        return e is not None and e <= trig
    status, truck = run(course, course.waypoint('bank_approach'), 'wall_center', {'speed': 0.6},
                        radius=r, stop=walls_end)
    if status != 'stopped':
        return 'no_trigger', truck
    lead = bank_lead(r, mission.get('axle_to_camera', 0.38), **entry)
    run(course, None, 'heading_hold', {'speed': 0.5, 'distance': lead}, radius=r, truck=truck)
    xs = []
    status, truck = run(course, None, 'arc_turn', {'direction': 'right', 'turn_deg': 180,
                                                   'radius': r, 'speed': 0.5,
                                                   'allow_kturn': False},
                        radius=r, truck=truck, stop=lambda t, s, ok: xs.append(t.x) and False)
    bx, _, _, hl, _ = course.bank_box
    if max(xs) + 0.05 > bx + hl:
        return 'off_plane', truck
    return status, truck


CHAINS = {'bank': bank_chain}


def _in_return_lane(course, truck):
    # Heading back west inside the 32" return lane at the pothole ramp.
    x0, y0, _ = course.waypoint('bank_approach')
    return abs(truck.y - (y0 - 1.495)) < 0.45 and math.cos(truck.yaw) < -0.9 and truck.x < x0 + 3.0


def _through_narrow(course, truck):
    gx, gy, gyaw = course.waypoint('gravel')
    dx, dy = truck.x - gx, truck.y - gy
    return dx * math.cos(gyaw) + dy * math.sin(gyaw) > -0.5 and math.cos(truck.yaw - gyaw) > 0.94


def _through_buckets(course, truck):
    hx, hy, hyaw = course.waypoint('hoops')
    dx, dy = truck.x - hx, truck.y - hy
    return dx * math.cos(hyaw) + dy * math.sin(hyaw) > -0.2


ENDS = {'bank': _in_return_lane, 'narrow': _through_narrow, 'buckets': _through_buckets}


def main(argv):
    radius = None
    for a in list(argv):
        if a.startswith('--radius='):
            radius = float(a.split('=', 1)[1])
            argv.remove(a)
    names = list(SCENARIOS) + list(CHAINS) if not argv or argv[0] == 'all' else argv
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
