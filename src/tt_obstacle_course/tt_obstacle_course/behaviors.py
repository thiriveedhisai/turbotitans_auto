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
CLIMB = ('ramp_up', 'bank')         # terrain states where a stall needs momentum, not caution


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
    axle_back: float = 0.23       # rear axle (turn centre) behind the lidar
    camera_ahead: float = 0.15    # odometry point (camera) ahead of the lidar

    def g(self, group, key, default):
        return self.gains.get(group, {}).get(key, default)

    def front(self, scan):
        return scan.corridor_clearance(self.half_width + self.sweep_margin, True,
                                       min_x=self.front_overhang, body_half=self.half_width)

    def rear(self, scan):
        return scan.corridor_clearance(self.half_width + self.sweep_margin, False,
                                       min_x=self.rear_overhang, body_half=self.half_width)


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

    def clear_kappa(self, ctx, kappa, need=0.45, any_side=False, open_m=None):
        """The controller may ask for an arc that clips a wall or hoop leg; the governor
        then stops us and backing off just repeats it. Use the curvature nearest the
        request whose swept band is clear for `need` m (alternatives: `open_m`): easing
        off on the same side, or (any_side) also bending the other way. None clear:
        keep the request."""
        if not ctx.lidar_ok or ctx.scan is None:
            return kappa
        k_max = self.cfg.steering.max_curvature
        kappa = max(-k_max, min(k_max, kappa))
        band = self.cfg.half_width + 0.08

        def clear(k, d):
            return ctx.scan.corridor_clearance(band, True, min_x=self.cfg.front_overhang,
                                               curvature=k, axle_back=self.cfg.axle_back,
                                               body_half=self.cfg.half_width) >= d
        if clear(kappa, need):
            return kappa
        if any_side:
            options = sorted((f * k_max for f in (-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1)),
                             key=lambda k: abs(k - kappa))
        else:
            options = [kappa * f for f in (0.75, 0.5, 0.25, 0.0)]
        for k in options:
            if clear(k, open_m or need):
                return k
        return kappa

    def step(self, ctx, dt):
        raise NotImplementedError


class Hold(Behavior):
    """Stopped, wheels straight. Used for idle-after-done and 'stop'."""
    name = 'stop'

    def step(self, ctx, dt):
        return Command(0.0, 0.0, DONE)


class HeadingHold(Behavior):
    """Drive at a fixed heading, optionally nudged to the corridor centre.

    params: speed, heading_offset_deg | heading_abs (+ abs_offset_deg), center_gain,
            center_max_width, distance, lateral_offset (m, + left of the start
            line), line_point [x, y] (follow the line through this odom point
            instead), past_point (done once this far past line_point)
    """
    name = 'heading_hold'

    def start(self, ctx):
        super().start(ctx)
        if 'heading_abs' in self.p:
            self.target = wrap_angle(float(self.p['heading_abs'])
                                     + float(self.p.get('abs_offset_deg', 0.0)) * DEG)
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
        if 'line_point' in self.p:
            # Threading a hoop: never steer the swept arc into a leg.
            kappa = self.clear_kappa(ctx, kappa, need=0.4, any_side=True)
        speed = float(self.p.get('speed', 0.5))
        # Turning onto the heading: full speed only once roughly lined up (the car-wash
        # sprint started 60 deg off line at 1.5 m/s and hit the lane wall). Measured
        # against the base heading: a lateral-offset tilt must not slow a ramp climb.
        slow = min(speed, float(self.p.get('turn_speed', 0.4)))
        off = abs(wrap_angle(self.target - ctx.odom.yaw))
        f = min(1.0, max(0.0, (off - 25 * DEG) / (30 * DEG)))
        return Command(speed - (speed - slow) * f, self.steer_for(kappa), status)


class WallCenter(Behavior):
    """Lane following: fit a line to each lidar wall, pure-pursuit the lane centre.

    The lane width is remembered, so a missing wall (gap, end of a bale) falls
    back to the other wall at half that width. No walls: hold the last heading.
    params: speed, follow (center|left|right), wall_target (one-wall distance),
            max_wall (ignore walls farther than this), offset (+ left of centre),
            lookahead (m from the rear axle; default grows with speed), distance
    """
    name = 'wall_center'

    def start(self, ctx):
        super().start(ctx)
        self.hold_yaw = ctx.odom.yaw
        self.width = None

    def lane_centre(self, ctx, speed):
        """(target_x, target_y) in the lidar frame, or None."""
        follow = self.p.get('follow', 'center')
        max_wall = float(self.p.get('max_wall', 1.0))
        target = float(self.p.get('wall_target', 0.4))
        look = float(self.p.get('lookahead', self.cfg.g('wall_center', 'lookahead', 0.0)))
        if look <= 0.0:
            look = min(1.4, max(0.7, 0.5 + 0.6 * speed))
        x_t = look - self.cfg.axle_back
        front = ctx.scan.corridor_clearance(self.cfg.half_width + 0.1, True,
                                            min_x=self.cfg.front_overhang,
                                            body_half=self.cfg.half_width)
        if front < look:
            # Lane bends (or we're mis-headed) and its wall is inside the lookahead:
            # aim into the free space instead of a centre line fitted on a stub.
            gap = ctx.scan.follow_gap(1.5, 0.2, 0.5, min_width=2 * self.cfg.half_width + 0.1)
            if gap is not None:
                return gap[1] * math.cos(gap[0]), gap[1] * math.sin(gap[0])
        walls = ctx.scan.lane_walls(max_half_width=max_wall,
                                    x_max=max(0.6, min(1.6, front - 0.15)))
        left = walls[1] if follow != 'right' else None
        right = walls[-1] if follow != 'left' else None
        at = lambda w: w[0] + w[1] * x_t + w[3] * x_t * x_t  # noqa: E731
        tilt = lambda w: math.sqrt(1 + (w[1] + 2 * w[3] * x_t) ** 2)  # noqa: E731
        if left and right:
            w = (left[0] - right[0]) / math.sqrt(1 + ((left[1] + right[1]) / 2) ** 2)
            if 0.3 < w < 2 * max_wall:
                self.width = w if self.width is None else 0.8 * self.width + 0.2 * w
            y = (at(left) + at(right)) / 2
        elif left:
            half = self.width / 2 if (follow == 'center' and self.width) else target
            y = at(left) - half * tilt(left)
        elif right:
            half = self.width / 2 if (follow == 'center' and self.width) else target
            y = at(right) + half * tilt(right)
        else:
            return None
        return x_t, y + float(self.p.get('offset', 0.0))

    def step(self, ctx, dt):
        speed = float(self.p.get('speed', 0.6))
        status = DONE if self.distance_done(ctx) else RUNNING
        tgt = self.lane_centre(ctx, speed) if (ctx.lidar_ok and ctx.scan is not None) else None
        if tgt is None:
            # No walls (e.g. on a ramp): keep the last good heading.
            kappa = 2.0 * wrap_angle(self.hold_yaw - ctx.odom.yaw)
            return Command(speed, self.steer_for(kappa), status, 'no_walls')
        self.hold_yaw = ctx.odom.yaw
        dx, dy = tgt[0] + self.cfg.axle_back, tgt[1]
        kappa = 2.0 * dy / (dx * dx + dy * dy)
        # In a bend every gentler arc runs into the outer wall: ease off only into open space.
        return Command(speed, self.steer_for(self.clear_kappa(ctx, kappa, open_m=0.9)), status)


class GapFollow(Behavior):
    """Reactive follow-the-gap with an optional preferred heading.

    params: speed, goal_heading_deg (relative to start) | goal_heading_abs |
            goal_point [x, y] (odom) + goal_radius (done when that close), distance,
            and any gap_follow gain (max_range, bubble_radius, free_range, min_width)
    """
    name = 'gap_follow'

    def _gain(self, key, default):
        return float(self.p.get(key, self.cfg.g('gap_follow', key, default)))

    def step(self, ctx, dt):
        speed = float(self.p.get('speed', 0.5))
        status = DONE if self.distance_done(ctx) else RUNNING
        goal = None
        if 'goal_point' in self.p:
            px, py = (float(v) for v in self.p['goal_point'])
            goal = wrap_angle(math.atan2(py - ctx.odom.y, px - ctx.odom.x) - ctx.odom.yaw)
            if math.hypot(px - ctx.odom.x, py - ctx.odom.y) < float(self.p.get('goal_radius', 0.0)):
                status = DONE
        elif 'goal_heading_abs' in self.p:
            goal = wrap_angle(float(self.p['goal_heading_abs']) - ctx.odom.yaw)
        elif 'goal_heading_deg' in self.p:
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
    params: turn_deg | heading_abs (absolute end heading), direction, radius,
            speed, allow_kturn,
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
        if 'heading_abs' in self.p:
            # Turn (in the given direction) until this absolute heading is reached.
            self.goal = (self.dir * wrap_angle(float(self.p['heading_abs']) - ctx.odom.yaw)) % (2 * math.pi)
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


class PlanDrive(Behavior):
    """Plan a forward/reverse full-lock path (Hybrid-A*) to a goal pose on lidar
    returns accumulated in the odom frame, and track it with pure pursuit on the rear
    axle. Replans when new returns block the path.

    params: goal [x, y, yaw] (camera pose, odom frame), speed, reverse_speed,
            goal_tol, margin (search window around start and goal), allow_reverse
    """
    name = 'plan_drive'
    self_recovers = True       # stalls are handled here by replanning, not by blind backing off

    def start(self, ctx):
        super().start(ctx)
        from tt_obstacle_course.planner import Footprint, ObstacleMap
        self.a = self.cfg.axle_back + self.cfg.camera_ahead
        self.map = ObstacleMap()
        self.fp_rear, self.fp_front = self.cfg.odom_to_rear - self.a, self.a + self.cfg.odom_to_front
        self.fp = Footprint(rear=self.fp_rear, front=self.fp_front, half_width=self.cfg.half_width)
        self.path = None
        self.i = 0
        self.fails = 0
        self.t_plan = -1e9
        self.t_check = 0.0
        self.stall_since = None
        self.candidates = []
        if 'goal' not in self.p and self.p.get('goal_from_gap') and ctx.scan is not None:
            gap = ctx.scan.follow_gap(2.0, 0.2, 0.6, min_width=2 * self.cfg.half_width + 0.1)
            if gap is not None:
                # Recovery: fan of candidate poses between straight ahead and the free gap;
                # in a lane barely wider than the truck one ray's goal is often in the wall.
                ang, reach = gap[0], min(gap[1], float(self.p['goal_from_gap']))
                o = ctx.odom
                self.candidates = []
                for f in (1.0, 0.75, 0.5, 0.25):
                    for d in (reach, 0.75 * reach, 1.0):
                        a = f * ang
                        lx = d * math.cos(a) - self.cfg.camera_ahead
                        gx, gy = o.to_odom(lx, d * math.sin(a))
                        self.candidates.append([gx, gy, wrap_angle(o.yaw + a)])
                self.p['goal'] = self.candidates.pop(0)
                self.p.setdefault('radius_factor', 1.0)
        if 'goal' not in self.p:
            self.goal = None
            return
        gx, gy, gyaw = (float(v) for v in self.p['goal'])
        self.set_goal(gx, gy, gyaw)
        # Virtual walls (e.g. drop-off edges the lidar can't see): [[x1, y1, x2, y2], ...].
        for x1, y1, x2, y2 in self.p.get('fence', []):
            n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 0.03))
            for k in range(n + 1):
                t = k / n
                self.map.cells.add((int((x1 + t * (x2 - x1)) // self.map.res),
                                    int((y1 + t * (y2 - y1)) // self.map.res)))

    def set_goal(self, gx, gy, gyaw):
        self.goal_cam = (gx, gy, gyaw)
        self.goal = (gx - self.a * math.cos(gyaw), gy - self.a * math.sin(gyaw), gyaw)

    def rear(self, ctx):
        o = ctx.odom
        return o.x - self.a * math.cos(o.yaw), o.y - self.a * math.sin(o.yaw), o.yaw

    def grid(self, rear):
        m = float(self.p.get('margin', 1.5))
        xs, ys = (rear[0], self.goal[0]), (rear[1], self.goal[1])
        return self.map.inflated(min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m, self.fp.radius)

    def replan(self, ctx, rear):
        from tt_obstacle_course.planner import Footprint, free_goal, plan
        self.t_plan = ctx.t
        # Plan a little wider than full lock so the tracker keeps steering authority.
        radius = self.cfg.steering.min_turn_radius * float(self.p.get('radius_factor', 1.1))
        reverse = bool(self.p.get('allow_reverse', True))
        grid = self.grid(rear)
        goal = free_goal(self.goal, grid, self.fp) or self.goal
        self.path = plan(rear, goal, grid, self.fp, radius, allow_reverse=reverse)
        if self.path is None:
            # Already inside the safety margin (e.g. nose at a wall end): retry with the
            # bare body so the search can back out.
            tight = Footprint(rear=self.fp_rear, front=self.fp_front,
                              half_width=self.cfg.half_width, margin=0.0)
            m = float(self.p.get('margin', 1.5))
            xs, ys = (rear[0], self.goal[0]), (rear[1], self.goal[1])
            grid = self.map.inflated(min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m, tight.radius)
            goal = free_goal(self.goal, grid, tight) or self.goal
            self.path = plan(rear, goal, grid, tight, radius, allow_reverse=reverse)
        self.i = 0
        if self.path is None and self.candidates:
            # Unreachable recovery goal: try the next candidate straight away.
            self.set_goal(*self.candidates.pop(0))
            self.t_plan = -1e9
            return
        if self.path is None:
            self.fails += 1

    def step(self, ctx, dt):
        o = ctx.odom
        if self.goal is None:
            return Command(0.0, 0.0, FAILED, 'no goal')
        if ctx.lidar_ok and ctx.scan is not None:
            lx = o.x - self.cfg.camera_ahead * math.cos(o.yaw)
            ly = o.y - self.cfg.camera_ahead * math.sin(o.yaw)
            self.map.add_scan(ctx.scan, lx, ly, o.yaw, t=ctx.t)
        gx, gy, gyaw = self.goal_cam
        tol, ytol = float(self.p.get('goal_tol', 0.25)), float(self.p.get('yaw_tol', 0.4))
        dist, dyaw = math.hypot(gx - o.x, gy - o.y), abs(wrap_angle(gyaw - o.yaw))
        if dist <= tol and dyaw <= ytol:
            return Command(0.0, 0.0, DONE)
        if ctx.t - self.start_t > float(self.p.get('max_time', 45.0)):
            return Command(0.0, 0.0, FAILED, 'timeout')
        rear = self.rear(ctx)
        if self.path is not None and self.i >= len(self.path) - 2:
            ex, ey, ea, ed = self.path[-1][:4]
            past = (ed or 1) * ((rear[0] - ex) * math.cos(ea) + (rear[1] - ey) * math.sin(ea))
            if past > -0.05:
                # End of the path but off the goal (tracking error): close enough is
                # done (the next step corrects the rest); otherwise plan again from
                # here rather than driving on past the end.
                if dist <= 2 * tol and dyaw <= ytol:
                    return Command(0.0, 0.0, DONE, 'near goal')
                self.path = None
        if self.path is None:
            if self.fails > int(self.p.get('max_fails', 4)):
                return Command(0.0, 0.0, FAILED, 'no path')
            if ctx.t - self.t_plan >= 1.0:
                self.replan(ctx, rear)
            if self.path is None:
                return Command(0.0, 0.0, RUNNING, 'planning')
        elif ctx.t - self.t_check >= 0.5:
            from tt_obstacle_course.planner import path_blocked
            self.t_check = ctx.t
            if path_blocked(self.path, self.grid(rear), self.fp, self.i + 1):
                self.replan(ctx, rear)
                if self.path is None:
                    return Command(0.0, 0.0, RUNNING, 'replanning')
        cmd = self.track(rear)
        # Slipped off the path (e.g. wider arc on the banked plane): the tracker would
        # chase a segment it already left, across fenced edges. Stop and replan.
        p = self.path[self.i]
        if math.hypot(p[0] - rear[0], p[1] - rear[1]) > float(self.p.get('max_deviation', 0.3)):
            self.path = None
            return Command(0.0, 0.0, RUNNING, 'off path: replanning')
        # Commanding motion but not moving (governor stop or contact): mark the cells
        # just past the bumper in the direction of travel as blocked and replan.
        if abs(cmd.speed) >= 0.1 and abs(o.speed) < 0.05:
            if self.stall_since is None:
                self.stall_since = ctx.t
            elif ctx.t - self.stall_since > 1.2:
                self.stall_since = None
                d = 1.0 if cmd.speed > 0 else -1.0
                reach = (self.fp_front if d > 0 else self.fp_rear) + 0.1
                for lat in (-0.08, 0.0, 0.08):
                    bx = rear[0] + d * reach * math.cos(rear[2]) - lat * math.sin(rear[2])
                    by = rear[1] + d * reach * math.sin(rear[2]) + lat * math.cos(rear[2])
                    self.map.cells.add((int(bx // self.map.res), int(by // self.map.res)))
                self.replan(ctx, rear)
                return Command(0.0, 0.0, RUNNING, 'stalled: replanning')
        else:
            self.stall_since = None
        return cmd

    def track(self, rear):
        path = self.path
        d = path[self.i][3] or 1
        end = self.i
        while end + 1 < len(path) and path[end + 1][3] == d:
            end += 1
        # Progress: nearest point on the current same-direction segment.
        best = min(range(self.i, end + 1),
                   key=lambda k: math.hypot(path[k][0] - rear[0], path[k][1] - rear[1]))
        self.i = best
        ex, ey = path[end][0], path[end][1]
        to_end = math.hypot(ex - rear[0], ey - rear[1])
        if self.i >= end - 1 and to_end < 0.15 and end + 1 < len(path):
            self.i = end + 1           # cusp: switch direction
            return Command(0.0, 0.0, RUNNING, 'cusp')
        # Path-following on the rear axle: the planned arc's curvature as feedforward,
        # plus lateral and heading feedback. Reversing is the same law in the frame
        # facing the direction of travel (curvature and lateral error change sign).
        k = min(self.i, len(path) - 2)
        x0, y0, a0 = path[k][:3]
        x1, y1, a1 = path[k + 1][:3]
        seg = math.hypot(x1 - x0, y1 - y0)
        kref = wrap_angle(a1 - a0) / max(seg, 1e-3) if path[k + 1][3] == d else 0.0
        e = -math.sin(a0) * (rear[0] - x0) + math.cos(a0) * (rear[1] - y0)
        epsi = wrap_angle(rear[2] - a0)
        ke = float(self.p.get('k_lat', 2.5))
        kh = float(self.p.get('k_head', 1.8))
        if d > 0:
            kappa = kref - ke * e - kh * epsi
            v = min(float(self.p.get('speed', 0.4)), 0.15 + to_end)
        else:
            kappa = -kref - ke * e + kh * epsi
            v = -min(float(self.p.get('reverse_speed', 0.3)), 0.15 + to_end)
        return Command(v, self.steer_for(kappa), RUNNING)


class BackOff(Behavior):
    """Reverse for a distance (stuck recovery), optionally with a steering angle. If the
    rear is blocked before it has moved, it tries the opposite steer once."""
    name = 'backoff'

    def step(self, ctx, dt):
        if self.travelled(ctx) >= float(self.p.get('distance', 0.3)):
            return Command(0.0, 0.0, DONE)
        if (ctx.scan is not None and ctx.lidar_ok
                and self.cfg.rear(ctx.scan) < self.cfg.rear_overhang + 0.12):
            if self.travelled(ctx) < 0.05 and not self.p.get('flipped') and self.p.get('steer'):
                self.p['steer'], self.p['flipped'] = -float(self.p['steer']), True
                return Command(0.0, self.p['steer'], RUNNING, 'rear blocked: other lock')
            return Command(0.0, 0.0, DONE, 'rear blocked')
        return Command(-abs(float(self.p.get('speed', 0.3))), float(self.p.get('steer', 0.0)),
                       RUNNING)


REGISTRY = {cls.name: cls for cls in
            (Hold, HeadingHold, WallCenter, GapFollow, KTurn, ArcTurn, PlanDrive, BackOff)}


def make_behavior(name, params, cfg):
    if name not in REGISTRY:
        raise KeyError(f'unknown behavior {name!r}; known: {sorted(REGISTRY)}')
    return REGISTRY[name](params, cfg)


class StuckMonitor:
    """Detects 'commanding motion but not moving' and runs a BackOff, backing off further
    on each retry; afterwards the next stretch is driven slowly."""

    def __init__(self, min_speed=0.05, time_s=1.5, backoff_distance=0.3, max_retries=3,
                 slow_speed=0.25, slow_distance=0.8):
        self.min_speed = min_speed
        self.time_s = time_s
        self.backoff_distance = backoff_distance
        self.max_retries = max_retries
        self.slow_speed = slow_speed
        self.slow_distance = slow_distance
        self.slow_until = -1.0
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
        # Separate stuck events far apart shouldn't add up to a failure. Net displacement,
        # not the odometer: backing off and creeping forward again is no progress.
        here = (ctx.odom.x, ctx.odom.y)
        if self.progress_from is None:
            self.progress_from = here
        elif math.hypot(here[0] - self.progress_from[0], here[1] - self.progress_from[1]) > 1.5:
            self.retries = 0
            self.progress_from = here
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
        if self.retries > self.max_retries:
            return False
        # Reverse with opposite lock: the nose swings toward where it was steering.
        steer = -cfg.steering.full_lock(cmd.steer) if abs(cmd.steer) > 0.05 else 0.0
        dist = self.backoff_distance * (1.0 + 0.5 * (self.retries - 1))
        if ctx.terrain in CLIMB:
            dist += 0.4                     # run-up for the next attempt
        self.backoff = BackOff({'distance': dist, 'steer': steer}, cfg)
        self.backoff.start(ctx)
        return True

    def limit(self, ctx, cmd):
        """Retry slowly: cap forward speed for a stretch after a backoff. Not on a climb:
        a stall at the ramp crest (low friction) needs momentum, 0.25 m/s never got over."""
        if (cmd.speed > self.slow_speed and ctx.odom.distance < self.slow_until
                and ctx.terrain not in CLIMB):
            cmd.speed = self.slow_speed
        return cmd

    def exhausted(self):
        return self.retries > self.max_retries

    def step(self, ctx, dt):
        cmd = self.backoff.step(ctx, dt)
        if cmd.status != RUNNING:
            self.backoff = None
            self.slow_until = ctx.odom.distance + self.slow_distance
            return Command(0.0, 0.0, RUNNING, 'recovered')
        return Command(cmd.speed, cmd.steer, RUNNING, 'stuck backoff')
