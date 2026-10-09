"""Driving behaviors for the obstacle course. Pure logic, no ROS imports.

Each behavior is stepped at a fixed rate with a Context and returns a Command.
Steering is produced as curvature and converted through SteeringModel, so the
vehicle's (currently poor) minimum turning radius is respected everywhere.
"""

import math
from dataclasses import dataclass, field

from tt_obstacle_course.geometry import wrap_angle

DEG = math.pi / 180.0

RUNNING = 'running'
DONE = 'done'
FAILED = 'failed'


@dataclass
class Command:
    speed: float = 0.0
    steer: float = 0.0
    status: str = RUNNING
    info: str = ''


@dataclass
class Context:
    t: float
    odom: object                 # OdomTracker
    scan: object = None          # ScanView or None
    lidar_ok: bool = True        # False when tilted: 2D scan hits floor/sky
    terrain: str = 'flat'


@dataclass
class Config:
    steering: object             # SteeringModel
    gains: dict = field(default_factory=dict)
    half_width: float = 0.10     # vehicle half width (m)
    sweep_margin: float = 0.15   # extra strip width for blocked checks while turning
    front_overhang: float = 0.20  # lidar to front bumper (m)
    rear_overhang: float = 0.30   # lidar to rear bumper (m)
    odom_to_front: float = 0.05   # odometry point (camera) to front bumper
    odom_to_rear: float = 0.45    # odometry point (camera) to rear bumper

    def g(self, group, key, default):
        return self.gains.get(group, {}).get(key, default)

    def front(self, scan):
        return scan.corridor_clearance(self.half_width + self.sweep_margin, True,
                                       min_x=self.front_overhang)

    def rear(self, scan):
        return scan.corridor_clearance(self.half_width + self.sweep_margin, False,
                                       min_x=self.rear_overhang)


def _direction(value):
    if isinstance(value, str):
        return 1.0 if value.lower().startswith('l') else -1.0
    return 1.0 if value >= 0 else -1.0


class Behavior:
    name = 'base'

    def __init__(self, params, cfg):
        self.p = dict(params)
        self.cfg = cfg
        self.start_yaw = None
        self.start_dist = 0.0
        self.start_t = 0.0

    def start(self, ctx):
        self.start_yaw = ctx.odom.yaw
        self.start_dist = ctx.odom.distance
        self.start_t = ctx.t

    def travelled(self, ctx):
        return ctx.odom.distance - self.start_dist

    def distance_done(self, ctx):
        d = self.p.get('distance')
        return d is not None and self.travelled(ctx) >= float(d)

    def steer_for(self, kappa):
        return self.cfg.steering.curvature_to_steer(kappa)

    def step(self, ctx, dt):
        raise NotImplementedError


class Hold(Behavior):
    """Stopped, wheels straight. Used for idle-after-done and 'stop'."""
    name = 'stop'

    def step(self, ctx, dt):
        return Command(0.0, 0.0, DONE)


class HeadingHold(Behavior):
    """Drive at a fixed heading, optionally nudged to the corridor centre.

    params: speed, heading_offset_deg | heading_abs, center_gain,
            center_max_width, distance, lateral_offset (m, + left of the start
            line), line_point [x, y] (follow the line through this odom point
            instead), past_point (done once this far past line_point)
    """
    name = 'heading_hold'

    def start(self, ctx):
        super().start(ctx)
        if 'heading_abs' in self.p:
            self.target = float(self.p['heading_abs'])
        else:
            self.target = wrap_angle(ctx.odom.yaw + float(self.p.get('heading_offset_deg', 0.0)) * DEG)
        self.origin = (ctx.odom.x, ctx.odom.y)
        if 'line_point' in self.p:
            # Follow an absolute line (e.g. through a hoop centre) instead of the start pose.
            self.origin = tuple(float(v) for v in self.p['line_point'])
            self.p.setdefault('lateral_offset', 0.0)

    def along(self, ctx):
        c, s = math.cos(self.target), math.sin(self.target)
        return c * (ctx.odom.x - self.origin[0]) + s * (ctx.odom.y - self.origin[1])

    def step(self, ctx, dt):
        kp = self.cfg.g('heading_hold', 'kp', 2.0)
        kd = self.cfg.g('heading_hold', 'kd', 0.1)
        target = self.target
        if 'lateral_offset' in self.p:
            # Shift sideways off the start line (+ left) using odometry only.
            c, s = math.cos(self.target), math.sin(self.target)
            lat = -s * (ctx.odom.x - self.origin[0]) + c * (ctx.odom.y - self.origin[1])
            e = float(self.p['lateral_offset']) - lat
            cap = float(self.p.get('max_approach_deg', 20.0)) * DEG
            target = wrap_angle(self.target + max(-cap, min(cap, 1.2 * e)))
        err = wrap_angle(target - ctx.odom.yaw)
        kappa = kp * err - kd * ctx.odom.yaw_rate
        gain = float(self.p.get('center_gain', 0.0))
        if gain > 0.0 and ctx.lidar_ok and ctx.scan is not None:
            dl = ctx.scan.side_distance(+1)
            dr = ctx.scan.side_distance(-1)
            if dl + dr < float(self.p.get('center_max_width', 1.5)):
                kappa += gain * (dl - dr) / 2.0
        status = DONE if self.distance_done(ctx) else RUNNING
        if 'past_point' in self.p and self.along(ctx) >= float(self.p['past_point']):
            status = DONE
        return Command(float(self.p.get('speed', 0.5)), self.steer_for(kappa), status)


class WallCenter(Behavior):
    """Centre between lidar walls; follow one wall if only one is seen.

    params: speed, wall_target, max_wall, distance, follow (center|left|right)
    """
    name = 'wall_center'

    def start(self, ctx):
        super().start(ctx)
        self.prev_err = None
        self.hold_yaw = ctx.odom.yaw

    def step(self, ctx, dt):
        speed = float(self.p.get('speed', 0.6))
        kp = self.cfg.g('wall_center', 'kp', 1.2)
        kd = self.cfg.g('wall_center', 'kd', 0.15)
        look = self.cfg.g('wall_center', 'lookahead', 0.8)
        max_wall = float(self.p.get('max_wall', 2.0))
        target = float(self.p.get('wall_target', 0.4))
        status = DONE if self.distance_done(ctx) else RUNNING

        left = right = None
        if ctx.lidar_ok and ctx.scan is not None:
            follow = self.p.get('follow', 'center')
            if follow != 'right':
                left = ctx.scan.wall_estimate(+1, lookahead=look, max_range=max_wall)
            if follow != 'left':
                right = ctx.scan.wall_estimate(-1, lookahead=look, max_range=max_wall)
        if left and right:
            err = (left[0] - right[0]) / 2.0
        elif left:
            err = left[0] - target
        elif right:
            err = target - right[0]
        else:
            # No walls (e.g. on a ramp): keep the last good heading.
            kappa = 2.0 * wrap_angle(self.hold_yaw - ctx.odom.yaw)
            self.prev_err = None
            return Command(speed, self.steer_for(kappa), status, 'no_walls')

        self.hold_yaw = ctx.odom.yaw
        derr = 0.0 if self.prev_err is None or dt <= 0 else (err - self.prev_err) / dt
        self.prev_err = err
        kappa = kp * err + kd * derr
        return Command(speed, self.steer_for(kappa), status)


class GapFollow(Behavior):
    """Reactive follow-the-gap with an optional preferred heading.

    params: speed, goal_heading_deg (relative to start), distance, and any
            gap_follow gain (max_range, bubble_radius, free_range, min_width)
    """
    name = 'gap_follow'

    def _gain(self, key, default):
        return float(self.p.get(key, self.cfg.g('gap_follow', key, default)))

    def step(self, ctx, dt):
        speed = float(self.p.get('speed', 0.5))
        status = DONE if self.distance_done(ctx) else RUNNING
        goal = None
        if 'goal_heading_deg' in self.p:
            goal = wrap_angle(self.start_yaw + float(self.p['goal_heading_deg']) * DEG - ctx.odom.yaw)
        if not ctx.lidar_ok or ctx.scan is None:
            kappa = 0.0 if goal is None else 2.0 * goal
            return Command(speed * 0.5, self.steer_for(kappa), status, 'no_lidar')
        target = ctx.scan.follow_gap(
            max_range=self._gain('max_range', 3.0),
            bubble_radius=self._gain('bubble_radius', 0.20),
            free_range=self._gain('free_range', 0.8),
            goal_angle=goal,
            goal_gain=self._gain('goal_heading_gain', 0.4),
            min_width=self._gain('min_width', 0.35))
        if target is None:
            return Command(0.0, 0.0, status, 'no_gap')
        angle, depth = target
        kappa = 2.0 * math.sin(angle) / max(depth, 0.5)
        return Command(speed, self.steer_for(kappa), status)


class KTurn(Behavior):
    """Multi-point turn: forward at full lock, reverse at opposite lock, repeat.

    params: turn_deg, direction (left/right), leg_speed, leg_max_distance,
            rev_leg_max_distance, front_clearance, rear_clearance,
            heading_tolerance_deg, max_legs,
            bounds: [u_min, u_max, v_min, v_max] box in the start frame (u ahead,
            v left) the bumpers must stay in; odometry only, for areas the
            lidar can't judge (the 8.5 deg bank)
    """
    name = 'k_turn'

    def start(self, ctx):
        super().start(ctx)
        self.dir = _direction(self.p.get('direction', 'left'))
        self.goal = abs(float(self.p.get('turn_deg', 90.0))) * DEG
        self.turned = 0.0
        self.prev_yaw = ctx.odom.yaw
        self.legs = 0
        self.leg = +1
        self.leg_start = ctx.odom.distance
        self.pause_until = ctx.t
        self.frame = (ctx.odom.x, ctx.odom.y, ctx.odom.yaw)
        self.bounds = self.p.get('bounds')
        if self._front_blocked(ctx):
            self._switch(ctx)

    def _get(self, key, default):
        return float(self.p.get(key, self.cfg.g('k_turn', key, default)))

    def _outside(self, ctx, forward):
        if not self.bounds:
            return False
        u0, u1, v0, v1 = (float(b) for b in self.bounds)
        x0, y0, yaw0 = self.frame
        h = ctx.odom.yaw
        reach = self.cfg.odom_to_front + 0.05 if forward else -(self.cfg.odom_to_rear + 0.05)
        for side in (-self.cfg.half_width, self.cfg.half_width):
            px = ctx.odom.x + reach * math.cos(h) - side * math.sin(h)
            py = ctx.odom.y + reach * math.sin(h) + side * math.cos(h)
            dx, dy = px - x0, py - y0
            u = dx * math.cos(yaw0) + dy * math.sin(yaw0)
            v = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
            if not (u0 <= u <= u1 and v0 <= v <= v1):
                return True
        return False

    def _front_blocked(self, ctx):
        if self._outside(ctx, True):
            return True
        if ctx.scan is None or not ctx.lidar_ok:
            return False
        return self.cfg.front(ctx.scan) < self._get('front_clearance', 0.35)

    def _rear_blocked(self, ctx):
        if self._outside(ctx, False):
            return True
        if ctx.scan is None or not ctx.lidar_ok:
            return False
        return self.cfg.rear(ctx.scan) < self._get('rear_clearance', 0.30)

    def _switch(self, ctx):
        self.leg = -self.leg
        self.legs += 1
        self.leg_start = ctx.odom.distance
        # Let the servo reach the new lock before driving (less tyre scrub).
        self.pause_until = ctx.t + 0.3

    @property
    def remaining(self):
        return self.goal - self.turned

    def step(self, ctx, dt):
        self.turned += self.dir * wrap_angle(ctx.odom.yaw - self.prev_yaw)
        self.prev_yaw = ctx.odom.yaw
        steer = self.cfg.steering.full_lock(self.dir * self.leg)
        if self.remaining <= self._get('heading_tolerance_deg', 10.0) * DEG:
            return Command(0.0, 0.0, DONE)
        if self.legs > int(self._get('max_legs', 8)):
            return Command(0.0, 0.0, FAILED, 'max_legs')
        if ctx.t < self.pause_until:
            return Command(0.0, steer, RUNNING, 'settling')
        leg_len = ctx.odom.distance - self.leg_start
        leg_max = self._get('leg_max_distance', 0.6)
        rev_max = self._get('rev_leg_max_distance', leg_max)
        if ((self.leg > 0 and (self._front_blocked(ctx) or leg_len >= leg_max))
                or (self.leg < 0 and (self._rear_blocked(ctx) or leg_len >= rev_max))):
            self._switch(ctx)
            return Command(0.0, self.cfg.steering.full_lock(self.dir * self.leg), RUNNING, 'settling')
        speed = self.leg * self._get('leg_speed', 0.4)
        return Command(speed, steer, RUNNING, f'leg{self.legs} {"fwd" if self.leg > 0 else "rev"}')


class ArcTurn(Behavior):
    """Constant-radius turn by heading change, with yaw-rate feedback.

    Falls back to a KTurn for the rest of the angle if the front gets blocked.
    params: turn_deg, direction, radius, speed, allow_kturn,
            outer_wall_target (lidar side correction, flat ground only),
            center_left + track_radius: hold a circle around a centre that lies
            center_left metres to the turn side of the start pose (helix:
            no visible walls, so odometry is the only lateral reference)
    """
    name = 'arc_turn'

    def start(self, ctx):
        super().start(ctx)
        self.dir = _direction(self.p.get('direction', 'left'))
        self.goal = abs(float(self.p.get('turn_deg', 90.0))) * DEG
        radius = float(self.p.get('radius', self.cfg.steering.min_turn_radius))
        if radius < self.cfg.steering.min_turn_radius:
            self.warning = (f'radius {radius:.2f} m < vehicle min '
                            f'{self.cfg.steering.min_turn_radius:.2f} m; using full lock')
        else:
            self.warning = ''
        self.kappa = self.dir / radius
        self.turned = 0.0
        self.prev_yaw = ctx.odom.yaw
        self.kturn = None
        self.centre = None
        if 'center_left' in self.p:
            d = float(self.p['center_left']) * self.dir
            self.centre = (ctx.odom.x - math.sin(ctx.odom.yaw) * d,
                           ctx.odom.y + math.cos(ctx.odom.yaw) * d)
            self.track_r = float(self.p.get('track_radius', radius))

    def circle_curvature(self, ctx):
        dx, dy = ctx.odom.x - self.centre[0], ctx.odom.y - self.centre[1]
        r = math.hypot(dx, dy)
        tangent = math.atan2(dy, dx) + self.dir * math.pi / 2
        e_r = r - self.track_r              # > 0: outside the circle, tighten
        e_h = wrap_angle(tangent - ctx.odom.yaw)
        return (self.dir / self.track_r + self.dir * self.cfg.g('arc_turn', 'radius_gain', 2.0) * e_r
                + self.cfg.g('arc_turn', 'tangent_gain', 1.5) * e_h)

    def step(self, ctx, dt):
        if self.kturn is not None:
            return self.kturn.step(ctx, dt)
        self.turned += self.dir * wrap_angle(ctx.odom.yaw - self.prev_yaw)
        self.prev_yaw = ctx.odom.yaw
        tol = self.cfg.g('arc_turn', 'tolerance_deg', 5.0) * DEG
        if self.turned >= self.goal - tol:
            return Command(0.0, 0.0, DONE)

        speed = float(self.p.get('speed', 0.5))
        if (self.p.get('allow_kturn', True) and ctx.lidar_ok and ctx.scan is not None
                and self.cfg.front(ctx.scan) < self.cfg.g('k_turn', 'front_clearance', 0.35)):
            rest = dict(self.p)
            rest['turn_deg'] = (self.goal - self.turned) / DEG
            self.kturn = KTurn(rest, self.cfg)
            self.kturn.start(ctx)
            return Command(0.0, 0.0, RUNNING, 'blocked -> k_turn')

        v = max(abs(ctx.odom.speed), 0.2)
        if self.centre is not None:
            return Command(speed, self.steer_for(self.circle_curvature(ctx)), RUNNING)
        kp = self.cfg.g('arc_turn', 'yaw_rate_kp', 0.8)
        kappa = self.kappa + kp * (self.kappa * v - ctx.odom.yaw_rate) / v
        outer = self.p.get('outer_wall_target')
        if outer is not None and ctx.lidar_ok and ctx.scan is not None:
            d = ctx.scan.side_distance(-self.dir)
            if math.isfinite(d):
                kappa += self.dir * self.cfg.g('arc_turn', 'edge_correction_gain', 0.5) * (float(outer) - d)
        return Command(speed, self.steer_for(kappa), RUNNING, self.warning)


class BackOff(Behavior):
    """Reverse for a distance (stuck recovery), optionally with a steering angle."""
    name = 'backoff'

    def step(self, ctx, dt):
        if self.travelled(ctx) >= float(self.p.get('distance', 0.3)):
            return Command(0.0, 0.0, DONE)
        if ctx.scan is not None and ctx.lidar_ok and self.cfg.rear(ctx.scan) < 0.2:
            return Command(0.0, 0.0, DONE, 'rear blocked')
        return Command(-abs(float(self.p.get('speed', 0.3))), float(self.p.get('steer', 0.0)),
                       RUNNING)


REGISTRY = {cls.name: cls for cls in
            (Hold, HeadingHold, WallCenter, GapFollow, KTurn, ArcTurn, BackOff)}


def make_behavior(name, params, cfg):
    if name not in REGISTRY:
        raise KeyError(f'unknown behavior {name!r}; known: {sorted(REGISTRY)}')
    return REGISTRY[name](params, cfg)


class StuckMonitor:
    """Detects 'commanding motion but not moving' and runs a BackOff."""

    def __init__(self, min_speed=0.05, time_s=1.5, backoff_distance=0.3, max_retries=3):
        self.min_speed = min_speed
        self.time_s = time_s
        self.backoff_distance = backoff_distance
        self.max_retries = max_retries
        self.reset()

    def reset(self):
        self.since = None
        self.retries = 0
        self.backoff = None
        self.progress_from = None

    def active(self):
        return self.backoff is not None

    def check(self, ctx, cmd, cfg):
        """Return True if a recovery backoff was started."""
        # Separate stuck events far apart shouldn't add up to a failure.
        if self.progress_from is None:
            self.progress_from = ctx.odom.distance
        elif ctx.odom.distance - self.progress_from > 2.0:
            self.retries = 0
            self.progress_from = ctx.odom.distance
        if abs(cmd.speed) < 0.1 or abs(ctx.odom.speed) >= self.min_speed:
            self.since = None
            return False
        if self.since is None:
            self.since = ctx.t
            return False
        if ctx.t - self.since < self.time_s:
            return False
        self.since = None
        self.retries += 1
        self.progress_from = ctx.odom.distance
        if self.retries > self.max_retries:
            return False
        # Reverse with opposite lock: the nose swings toward where it was steering.
        steer = -cfg.steering.full_lock(cmd.steer) if abs(cmd.steer) > 0.05 else 0.0
        self.backoff = BackOff({'distance': self.backoff_distance, 'steer': steer}, cfg)
        self.backoff.start(ctx)
        return True

    def exhausted(self):
        return self.retries > self.max_retries

    def step(self, ctx, dt):
        cmd = self.backoff.step(ctx, dt)
        if cmd.status != RUNNING:
            self.backoff = None
            return Command(0.0, 0.0, RUNNING, 'recovered')
        return Command(cmd.speed, cmd.steer, RUNNING, 'stuck backoff')
