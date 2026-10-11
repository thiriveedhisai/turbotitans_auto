"""Mission manager: walks the course zone by zone from config/mission.yaml.

Each zone is a list of steps. A step is one of:
  behavior: <name>  params: {...}               -> behavior_executor drives
  nav2: {goal: relative|gap, ...}  fallback: {behavior, params}
  gate: {approach, beyond, search_forward, ...}  (hoops: Nav2 to a pre-gate
        pose, then a straight blind commit through)
A step ends when any 'until' trigger fires (default: done):
  done | failed | terrain=X | terrain!=X | left=X | marker=X | distance>=m | time>=s |
  rise>=m | yaw>=deg | gap=left|right | walls_lost>=m | open_left>=m | open_right>=m |
  walls_end<=m | opening_ahead>=m |
  corridor<=m | exit_seen=lead | finish_funnel<=m
Guards: after_distance (m), min_time (s). Zone timeout_s moves on regardless.
Seeing a marker for a zone up to marker_skip_max ahead jumps straight to it.
"""

import json
import math
import re

import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

from tt_obstacle_course.geometry import (OdomTracker, quat_to_rpy, wrap_angle,
                                         yaw_to_quat)
from tt_obstacle_course.ros_common import RELIABLE_SUB, TTNode, dumps, odom_update
from tt_obstacle_course.ros_common import run as run_node
from tt_obstacle_course.scan_utils import ScanView

TRIGGER = re.compile(r'^\s*(\w+)\s*(>=|<=|!=|=)?\s*([\w.\-]+)?\s*$')
NO_CAP = 99.0


def parse_trigger(text):
    m = TRIGGER.match(str(text))
    if not m:
        raise ValueError(f'bad trigger {text!r}')
    return m.group(1), m.group(2), m.group(3)


class Nav2Link:
    """Thin NavigateToPose client that tolerates goals being replaced."""

    def __init__(self, node, action_name, frame):
        self.node = node
        self.frame = frame
        self.client = ActionClient(node, NavigateToPose, action_name)
        self.handle = None
        self.state = 'none'     # none | pending | active | succeeded | failed
        self.seq = 0

    def ready(self):
        return self.client.server_is_ready()

    def send(self, x, y, yaw):
        self.cancel()
        seq = self.seq
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.frame
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        o = goal.pose.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        self.state = 'pending'
        self.client.send_goal_async(goal).add_done_callback(lambda f: self._accepted(f, seq))
        self.node.get_logger().info(f'nav2 goal ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)')

    def _accepted(self, fut, seq):
        handle = fut.result()
        if seq != self.seq:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.state = 'failed'
            return
        self.handle = handle
        self.state = 'active'
        handle.get_result_async().add_done_callback(lambda f: self._result(f, seq))

    def _result(self, fut, seq):
        if seq != self.seq:
            return
        self.handle = None
        ok = fut.result().status == GoalStatus.STATUS_SUCCEEDED
        self.state = 'succeeded' if ok else 'failed'

    def cancel(self):
        self.seq += 1
        if self.handle is not None:
            self.handle.cancel_goal_async()
            self.handle = None
        self.state = 'none'


class StepRun:

    def __init__(self, spec, t0, odom):
        self.spec = spec
        self.t0 = t0
        self.d0 = odom.distance
        self.z0 = odom.z
        self.pose0 = (odom.x, odom.y, odom.yaw)
        self.prev_yaw = odom.yaw
        self.turned = 0.0
        self.phase = None
        self.done = False
        self.failed = False
        self.fallback = False
        self.searches = 0
        self.target = None
        self.t_reaim = 0.0
        self.terrain_seen = set()
        self.queue = []


class MissionManager(TTNode):

    def __init__(self):
        super().__init__('mission_manager')
        path = self.param('mission_file', '')
        if not path:
            raise RuntimeError('mission_file parameter is required')
        with open(path, encoding='utf-8') as f:
            self.mission = yaml.safe_load(f)
        self.zones = self.mission['zones']
        self.zone_index = {z['name']: i for i, z in enumerate(self.zones)}
        for z in self.zones:
            for s in z.get('steps', []):
                for t in s.get('until', ['done']):
                    parse_trigger(t)
        self.skip_max = int(self.mission.get('marker_skip_max', 2))
        self.skip_min_time = float(self.mission.get('marker_skip_min_time_s', 2.0))
        self.marker_lead = float(self.mission.get('marker_lead_m', 1.3))
        self.pending_markers = {}
        self.use_nav2 = bool(self.param('use_nav2', True))
        self.variables = self.compute_variables()
        # Begin at a later zone (practice a section, or resume after a reset).
        self.start_zone = str(self.param('start_zone', ''))
        self.yaw_offset, self.flip = self.scan_args()

        self.odom = OdomTracker()
        self.nav = Nav2Link(self, self.param('nav2_action', 'navigate_to_pose'),
                            self.param('odom_frame', 'odom'))
        self.started = False
        self.race_t0 = None
        self.zone_idx = -1
        self.zone_t0 = 0.0
        self.step_idx = -1
        self.run = None
        self.cmd_id = 0
        self.cmd = None
        self.bstatus = {}
        self.terrain = 'flat'
        self.lidar_ok = True
        self.scan = None
        self.markers = {}
        self.gaps = {}
        self.gate = None
        self.gate_confirmed = False
        self.passed_gates = []
        self.last_gap = None
        self.lane_origin = None
        self.last_hoop_along = 0.0
        # Bucket-field exit -> start of the hoop lane (36" connector, ~0.83 m).
        self.connector = float(self.mission.get('hoop_lane', {}).get('connector', 0.83))
        self.finished = False

        self.pub_cmd = self.create_publisher(String, '/behavior/cmd', 10)
        self.pub_state = self.create_publisher(String, '/mission/state', 10)
        self.pub_cap = self.create_publisher(Float32, '/mission/speed_cap', 10)
        self.create_subscription(Bool, '/race_start', self.on_start, RELIABLE_SUB)
        self.create_subscription(String, '/zone_event', self.on_marker, 10)
        self.create_subscription(String, '/terrain/state', self.on_terrain, 10)
        self.create_subscription(Bool, '/terrain/lidar_ok', self.on_lidar_ok, 10)
        self.create_subscription(String, '/behavior/status', self.on_bstatus, 10)
        self.create_subscription(Odometry, self.param('odom_topic', '/zed/zed_node/odom'),
                                 lambda m: odom_update(self.odom, m), 10)
        self.create_subscription(LaserScan, self.param('scan_topic', '/scan'), self.on_scan,
                                 qos_profile_sensor_data)
        for side in ('left', 'right', 'front'):
            self.create_subscription(PoseStamped, f'/perception/{side}_gap',
                                     lambda m, s=side: self.on_gap(s, m), 10)
        self.create_subscription(PoseStamped, '/perception/gate', self.on_gate, 10)
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.republish)
        self.send_behavior('stop', {})
        self.get_logger().info(f'Mission loaded: {[z["name"] for z in self.zones]}')

    # -------------------------------------------------------------- inputs
    def on_start(self, msg):
        if msg.data and not self.started:
            self.started = True
            self.race_t0 = self.now_s()
            self.get_logger().info('RACE START')

    def on_marker(self, msg):
        # Markers are seen up to trigger_range ahead; act once we've driven there.
        if msg.data not in self.pending_markers and self.odom.valid:
            self.pending_markers[msg.data] = self.odom.distance + self.marker_lead

    def process_markers(self):
        for zone, at in list(self.pending_markers.items()):
            if self.odom.distance >= at:
                del self.pending_markers[zone]
                self.apply_marker(zone)

    def apply_marker(self, zone):
        now = self.now_s()
        self.markers[zone] = now
        j = self.zone_index.get(zone)
        # Never cut a Nav2/gate manoeuvre short; the step's own triggers finish it.
        busy = self.run is not None and ('nav2' in self.run.spec or 'gate' in self.run.spec) \
            and not self.run.fallback
        if (self.started and not self.finished and j is not None and self.zone_idx >= 0
                and not busy
                and 0 < j - self.zone_idx <= self.skip_max
                and now - self.zone_t0 >= self.skip_min_time
                and self.zones[j].get('allow_marker_skip', True)):
            self.get_logger().warn(f'marker for {zone}: jumping from {self.zone["name"]}')
            self.enter_zone(j)

    def on_terrain(self, msg):
        self.terrain = msg.data

    def on_lidar_ok(self, msg):
        self.lidar_ok = msg.data

    def on_bstatus(self, msg):
        try:
            s = json.loads(msg.data)
        except ValueError:
            return
        self.bstatus[s.get('id')] = s.get('status')

    def on_scan(self, msg):
        self.scan = ScanView.from_msg(msg, self.yaw_offset, self.flip)

    @staticmethod
    def pose_of(msg):
        o = msg.pose.orientation
        return msg.pose.position.x, msg.pose.position.y, quat_to_rpy(o.x, o.y, o.z, o.w)[2]

    def on_gap(self, side, msg):
        self.gaps[side] = (self.pose_of(msg), self.now_s())

    def on_gate(self, msg):
        self.gate = (self.pose_of(msg), self.now_s())
        self.gate_confirmed = msg.pose.position.z > 0.5

    # ------------------------------------------------------------ commands
    def compute_variables(self):
        """Values mission.yaml can reference as "$name" / "-$name"."""
        radius = float(self.param('vehicle.planning_min_turn_radius', 1.52))
        h = self.mission.get('helix_entry', {})
        axle = float(self.mission.get('axle_to_camera', 0.38))
        # The camera (odometry point) sits ahead of the rear axle, so it circles
        # wider than the turning radius; shift outward to stay concentric.
        cam_r = math.hypot(radius, axle)
        offset = cam_r - float(h.get('centerline', 1.22)) - float(h.get('margin', 0.05))
        offset = max(0.0, min(float(h.get('max_offset', 0.25)), offset))
        values = {'helix_offset': round(offset, 3), 'turn_radius': radius, 'axle_to_camera': axle}
        # Side-gap turn-in: follow the wall this far out so a full-lock 90 deg turn ends
        # end_clear before the wall line, square to the opening.
        end_clear = float(self.mission.get('gap_entry', {}).get('end_clear', 0.3))
        values['turn_in_offset'] = round(radius + axle + end_clear, 3)
        self.get_logger().info(f'turning radius {radius:.2f} m -> {values}')
        return values

    def resolve(self, params):
        out = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith('$exit') and 'exit_point' not in self.variables:
                self.compute_exit_point()
            if isinstance(v, str) and v.lstrip('-').startswith('$'):
                sign = -1.0 if v.startswith('-') else 1.0
                name = v.lstrip('-')[1:]
                if name not in self.variables:
                    self.get_logger().warn(f'{name} not known yet; dropping param {k}')
                    continue
                v = self.variables[name]
                if sign < 0:
                    v = [-x for x in v] if isinstance(v, list) else -v
            out[k] = v
        return out

    def valid_exit(self, det):
        """A front opening that can be the bucket-field exit: it must face the same way
        as the entrance and lie at the far wall's depth (rejects gaps between buckets
        and field corners). The far wall is square to the entrance, so the axis is the
        entrance's (an obliquely seen opening's edge-to-edge normal is not)."""
        ex, ey, eyaw = det
        if self.last_gap is None:
            return det
        gx, gy, gyaw = self.last_gap
        depth = math.cos(gyaw) * (ex - gx) + math.sin(gyaw) * (ey - gy)
        hint = float(self.mission.get('exit_hint', {}).get('depth', 3.67))
        if abs(wrap_angle(eyaw - gyaw)) > math.radians(60) or abs(depth - hint) > 0.8:
            return None
        # Pin it to the far wall: up close the opening's measured edges slide into the
        # connector beyond, but the field depth is fixed; only the lateral position moves.
        c, s = math.cos(gyaw), math.sin(gyaw)
        lat = -s * (ex - gx) + c * (ey - gy)
        return gx + c * hint - s * lat, gy + s * hint + c * lat, gyaw

    def track_exit(self):
        """In the bucket field: keep the exit estimate on the latest validated detection,
        re-sending an active plan to it if it moved."""
        g = self.gaps.get('front')
        if g is None or self.now_s() - g[1] > 0.5:
            return
        det = self.valid_exit(g[0])
        if det is None:
            return
        old = self.variables.get('exit_point')
        if old is not None and math.hypot(det[0] - old[0], det[1] - old[1]) < 0.2:
            return
        self.set_exit(det, 0.8)
        spec = self.run.spec if self.run else {}
        if spec.get('params', {}).get('goal') == '$exit_goal':
            self.send_behavior(spec['behavior'], spec['params'])

    def set_exit(self, gap, lead):
        """Waypoints through a detected exit: `lead` m before it on its axis, then past it."""
        gx, gy, gyaw = gap
        c, s = math.cos(gyaw), math.sin(gyaw)
        self.variables['exit_point'] = [gx, gy]
        self.variables['exit_approach'] = [gx - c * lead, gy - s * lead]
        self.variables['exit_through'] = [gx + c * 0.6, gy + s * 0.6]
        self.variables['exit_goal'] = [gx + c * 0.8, gy + s * 0.8, gyaw]
        self.variables['exit_yaw'] = gyaw
        self.set_lane_origin(gx + c * self.connector, gy + s * self.connector, gyaw)
        self.get_logger().info(f'exit seen at ({gx:.2f}, {gy:.2f}), axis {math.degrees(gyaw):.0f} deg')

    def set_lane_origin(self, x, y, yaw):
        """Start of the 72" hoop lane: hoop lines are measured from here."""
        self.lane_origin = (x, y, yaw)
        self.last_hoop_along = 0.0

    def reanchor_lane(self, expect, gate):
        """Shift the lane origin along the lane so the hoop just passed sits on its expected
        line: later hoops are then measured from a fresh sighting, not odometry since the exit."""
        if not expect or 'along' not in expect or self.lane_origin is None:
            return
        ox, oy, oyaw = self.lane_origin
        c, s = math.cos(oyaw), math.sin(oyaw)
        shift = c * (gate[0] - ox) + s * (gate[1] - oy) - float(expect['along'])
        self.lane_origin = (ox + c * shift, oy + s * shift, oyaw)
        self.last_hoop_along = float(expect['along'])
        self.get_logger().info(f'hoop lane re-anchored by {shift:+.2f} m')

    def set_bank_frame(self, wall_end):
        """The lane walls end at the bank plane's near edge, `wall_end` ahead of the lidar.
        From the fixed plane size, fence its far and side edges (drop-offs the lidar can't
        see) and set a goal in the return lane, for the U-turn planner."""
        b = self.mission.get('bank_entry', {})
        o = self.odom
        c, s = math.cos(o.yaw), math.sin(o.yaw)
        w = lambda u, v: [o.x + c * u - s * v, o.y + s * u + c * v]  # noqa: E731
        near = wall_end - float(b.get('lidar_behind_camera', 0.15))
        far = near + float(b.get('plane_depth', 1.219)) - float(b.get('margin', 0.15))
        north, south = float(b.get('north', 0.80)), float(b.get('south', -2.40))
        self.variables['bank_fence'] = [w(far, north) + w(far, south), w(near, north) + w(far, north),
                                        w(near, south) + w(far, south)]
        self.variables['bank_goal'] = w(near - 0.6, float(b.get('return_offset', -1.73))) + \
            [wrap_angle(o.yaw + math.pi)]
        self.get_logger().info(f'bank plane edge {near:.2f} m ahead; U-turn goal {self.variables["bank_goal"]}')

    def square(self, yaw):
        """Snap a measured lane axis to the rectilinear course grid (odom heading is kept
        on-grid by odom_guard's wall/tag corrections) if within square_axes_deg."""
        tol = math.radians(float(self.mission.get('square_axes_deg', 0.0)))
        grid = round(yaw / (math.pi / 2)) * (math.pi / 2)
        return wrap_angle(grid) if abs(wrap_angle(yaw - grid)) <= tol else yaw

    def set_entrance(self, gap, side=None):
        """Remember the opening just entered; a probe goal inside the field lets the
        planner start before the exit is seen. A side opening is cut square into the wall
        we're following: its axis is our heading +-90 deg on the grid (edge-based axes of
        an obliquely seen opening were up to 45 deg off)."""
        gx, gy, gyaw = gap
        if side in ('left', 'right'):
            turn = math.pi / 2 if side == 'left' else -math.pi / 2
            grid = round((self.odom.yaw + turn) / (math.pi / 2)) * (math.pi / 2)
            gyaw = wrap_angle(grid)
        else:
            gyaw = self.square(gyaw)
        gap = (gx, gy, gyaw)
        self.last_gap = gap
        self.variables['gap_yaw'] = gyaw
        d = float(self.mission.get('exit_hint', {}).get('depth', 3.67)) * 0.6
        self.variables['field_probe'] = [gx + d * math.cos(gyaw), gy + d * math.sin(gyaw), gyaw]

    def compute_exit_point(self):
        """Bucket-field exit from fixed field geometry: buckets and the entrance
        move between runs, the field walls and exit don't. Measured from the
        entrance gap along its axis, and from the lidar-measured side wall."""
        if self.last_gap is None:
            return
        hint = self.mission.get('exit_hint', {})
        gx, gy, gyaw = self.last_gap
        c, s = math.cos(gyaw), math.sin(gyaw)
        depth = float(hint.get('depth', 4.0))
        side = -1 if hint.get('wall', 'right') == 'right' else 1
        lateral = 0.0
        wall = None
        if self.scan is not None and self.lidar_ok:
            wall = self.scan.far_side_wall(side)
        if wall is not None:
            rx, ry = self.odom.x - gx, self.odom.y - gy
            robot_lat = -s * rx + c * ry
            wall_lat = robot_lat + side * wall
            lateral = wall_lat - side * float(hint.get('from_wall', 1.05))
        point = [gx + c * depth - s * lateral, gy + s * depth + c * lateral]
        self.set_exit((point[0], point[1], gyaw), float(hint.get('approach', 0.8)))
        self.get_logger().info(f'bucket exit estimate {point[0]:.2f}, {point[1]:.2f} '
                               f'(side wall {"%.2f m" % wall if wall else "unseen"})')

    @property
    def zone(self):
        return self.zones[self.zone_idx]

    def send_behavior(self, name, params):
        self.cmd_id += 1
        self.cmd = {'id': self.cmd_id, 'behavior': name, 'params': self.resolve(params)}
        self.pub_cmd.publish(String(data=dumps(self.cmd)))

    def republish(self):
        if self.cmd is not None:
            self.pub_cmd.publish(String(data=dumps(self.cmd)))
        self.publish_state()

    def speed_cap(self):
        if not self.started or self.finished or self.zone_idx < 0:
            return 0.0
        spec = self.run.spec if self.run else {}
        return float(spec.get('speed_cap', self.zone.get('speed_cap', NO_CAP)))

    def publish_state(self):
        state = {'started': self.started, 'finished': self.finished,
                 'zone': self.zone['name'] if self.zone_idx >= 0 else None,
                 'step': self.step_idx, 'phase': self.run.phase if self.run else None,
                 'elapsed': round(self.now_s() - self.race_t0, 1) if self.race_t0 else 0.0}
        self.pub_state.publish(String(data=dumps(state)))

    def nav_usable(self):
        return self.use_nav2 and self.nav.ready()

    # --------------------------------------------------------- transitions
    def enter_zone(self, i):
        self.nav.cancel()
        if i >= len(self.zones):
            self.finish()
            return
        self.zone_idx = i
        self.zone_t0 = self.now_s()
        self.step_idx = -1
        self.get_logger().info(f'=== ZONE {self.zone["name"]} '
                               f'(t={self.zone_t0 - self.race_t0:.1f} s) ===')
        self.next_step()

    def next_step(self, prev_failed=False):
        self.nav.cancel()
        self.step_idx += 1
        steps = self.zone.get('steps', [])
        # Recovery blocks (only_if_failed) run only when the step before the block failed.
        prev_recovery = 0 < self.step_idx <= len(steps) and \
            steps[self.step_idx - 1].get('only_if_failed', False)
        if not (prev_failed or prev_recovery):
            while self.step_idx < len(steps) and steps[self.step_idx].get('only_if_failed'):
                self.step_idx += 1
        if self.step_idx >= len(steps):
            self.enter_zone(self.zone_idx + 1)
            return
        spec = steps[self.step_idx]
        self.run = StepRun(spec, self.now_s(), self.odom)
        self.get_logger().info(f'step {self.step_idx}: {self.describe(spec)}')
        if 'behavior' in spec:
            self.send_behavior(spec['behavior'], spec.get('params', {}))
        elif 'nav2' in spec or 'gate' in spec:
            own_planner = spec.get('nav2', {}).get('entry') == 'plan'
            if not self.nav_usable() and 'gate' not in spec and not own_planner:
                self.start_fallback('nav2 not available')
            else:
                self.send_behavior('idle', spec.get('params', {}))
        else:
            self.get_logger().error(f'step without behavior/nav2/gate: {spec}')
            self.next_step()

    def start_fallback(self, reason):
        fb = self.run.spec.get('fallback')
        self.nav.cancel()
        if fb is None:
            self.get_logger().warn(f'{reason}; no fallback, skipping step')
            self.run.failed = True
            return
        self.get_logger().warn(f'{reason}; fallback -> {fb["behavior"]}')
        self.run.fallback = True
        self.send_behavior(fb['behavior'], fb.get('params', {}))

    def finish(self):
        if not self.finished:
            self.finished = True
            self.send_behavior('stop', {})
            self.get_logger().info(f'FINISHED in {self.now_s() - self.race_t0:.1f} s')

    @staticmethod
    def describe(spec):
        for key in ('behavior', 'nav2', 'gate'):
            if key in spec:
                return f'{key}={spec[key]} until={spec.get("until", ["done"])}'
        return str(spec)

    # ------------------------------------------------------------ triggers
    def fired(self, trigger, now):
        kind, op, val = parse_trigger(trigger)
        run = self.run
        bstat = self.bstatus.get(self.cmd_id)
        if kind == 'done':
            if 'behavior' in run.spec or run.fallback:
                return bstat == 'done'
            return run.done
        if kind == 'failed':
            return run.failed or bstat == 'failed'
        if kind == 'terrain':
            return (self.terrain == val) if op == '=' else (self.terrain != val)
        if kind == 'left':
            # Was in terrain state X during this step and no longer is.
            return val in run.terrain_seen and self.terrain != val
        if kind == 'marker':
            return self.markers.get(val, -1e9) >= run.t0
        if kind == 'distance':
            return self.odom.distance - run.d0 >= float(val)
        if kind == 'rise':
            return self.odom.z - run.z0 >= float(val)
        if kind == 'time':
            return now - run.t0 >= float(val)
        if kind == 'yaw':
            return abs(math.degrees(run.turned)) >= float(val)
        if kind == 'gap':
            g = self.gaps.get(val)
            return g is not None and g[1] >= run.t0
        if kind in ('open_left', 'open_right'):
            # That side's wall has ended (lane opens into an area): lidar, not odometry.
            if not self.lidar_ok or self.scan is None:
                return False
            return self.scan.side_distance(+1 if kind == 'open_left' else -1) >= float(val)
        if kind == 'walls_lost':
            if not self.lidar_ok or self.scan is None:
                return False
            return min(self.scan.side_distance(+1), self.scan.side_distance(-1)) >= float(val)
        if kind == 'opening_ahead':
            # Both walls end just ahead: the +/-60 deg beams stop hitting them.
            if not self.lidar_ok or self.scan is None:
                return False
            return min(self.scan.range_at(math.radians(60)),
                       self.scan.range_at(math.radians(-60))) >= float(val)
        if kind == 'walls_end':
            # Both side walls stop within val metres ahead of the lidar.
            if not self.lidar_ok or self.scan is None:
                return False
            end = self.scan.walls_end()
            if end is not None and end <= float(val):
                self.set_bank_frame(end)
                return True
            return False
        if kind == 'exit_seen':
            # Opening in the far wall confirmed by perception since this step began.
            g = self.gaps.get('front')
            return g is not None and g[1] >= run.t0 and self.valid_exit(g[0]) is not None
        if kind == 'finish_funnel':
            # Lane narrows to <= val with clear space ahead: out of the ribbons, at the
            # 48" -> 32" funnel (ribbons alone read as walls but also block the front).
            if not self.lidar_ok or self.scan is None:
                return False
            width = self.scan.side_distance(+1) + self.scan.side_distance(-1)
            return width <= float(val) and self.scan.corridor_clearance(0.2, True) >= 2.0
        if kind == 'corridor':
            # Walls on both sides closer together than val (e.g. an exit lane).
            if not self.lidar_ok or self.scan is None:
                return False
            return self.scan.side_distance(+1) + self.scan.side_distance(-1) <= float(val)
        raise ValueError(f'unknown trigger {trigger!r}')

    def step_finished(self, now):
        spec, run = self.run.spec, self.run
        if self.odom.distance - run.d0 < float(spec.get('after_distance', 0.0)):
            return None
        if now - run.t0 < float(spec.get('min_time', 0.0)):
            return None
        if run.failed or self.bstatus.get(self.cmd_id) == 'failed':
            return 'failed'
        for trig in spec.get('until', ['done']):
            if self.fired(trig, now):
                return trig
        if now - run.t0 > float(spec.get('timeout_s', 1e9)):
            return 'step timeout'
        return None

    # ---------------------------------------------------------------- loop
    def tick(self):
        self.pub_cap.publish(Float32(data=self.speed_cap()))
        if not self.started or self.finished or not self.odom.valid:
            return
        now = self.now_s()
        if self.zone_idx < 0:
            # Started mid-course: the start pose stands in for the entrance just passed.
            if self.start_zone and self.last_gap is None:
                self.set_entrance((self.odom.x, self.odom.y, self.odom.yaw))
                o = self.odom
                ahead = self.connector + 0.3     # hoops test start sits 0.3 m before the exit
                self.set_lane_origin(o.x + ahead * math.cos(o.yaw), o.y + ahead * math.sin(o.yaw), o.yaw)
            self.variables.setdefault('gap_yaw', self.odom.yaw)
            self.enter_zone(self.zone_index.get(self.start_zone, 0))
            return
        self.process_markers()
        if self.finished:
            return
        if self.zone['name'] == 'buckets':
            self.track_exit()
        run = self.run
        run.terrain_seen.add(self.terrain)
        run.turned += wrap_angle(self.odom.yaw - run.prev_yaw)
        run.prev_yaw = self.odom.yaw

        if now - self.zone_t0 > float(self.zone.get('timeout_s', 1e9)):
            self.get_logger().warn(f'zone {self.zone["name"]} timed out')
            self.enter_zone(self.zone_idx + 1)
            return
        if not run.fallback:
            if 'nav2' in run.spec:
                limit = run.spec['nav2'].get('nav_timeout_s')
                if limit is not None and run.phase not in ('search', 'turn_in', 'plan') \
                        and now - run.t0 > float(limit):
                    self.nav_failed(run, f'goal not reached in {float(limit):.0f} s')
                else:
                    self.update_nav(run)
            elif 'gate' in run.spec:
                self.update_gate(run, now)
        elif run.spec.get('nav2', {}).get('goal') == 'gap':
            self.update_gap_fallback(run)
        why = self.step_finished(now)
        if why:
            self.get_logger().info(f'step {self.step_idx} ended: {why}')
            self.next_step(prev_failed=why in ('failed', 'step timeout'))

    # ------------------------------------------------------------- nav2 steps
    def relative(self, base, forward, left, yaw_deg=0.0):
        x, y, yaw = base
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * forward - s * left, y + s * forward + c * left,
                wrap_angle(yaw + math.radians(yaw_deg)))

    def nav_failed(self, run, what):
        if run.spec.get('fallback'):
            self.start_fallback(f'nav2 {what} failed')
        else:
            self.get_logger().warn(f'nav2 {what} failed')
            run.failed = True

    def fresh_gap(self, run, side, min_ahead, min_from_start=None):
        gap = self.gaps.get(side)
        if gap is None or gap[1] < run.t0 or self.now_s() - gap[1] > 1.0:
            return None
        gx, gy, _ = gap[0]
        if self.odom.to_robot(gx, gy)[0] <= min_ahead:
            return None
        if min_from_start is not None:
            # Ahead of where the search began, not of a wandering heading (seed 5 turned
            # east while wall-following and took the pothole lane mouth for the entrance).
            sx, sy, syaw = run.pose0
            if math.cos(syaw) * (gx - sx) + math.sin(syaw) * (gy - sy) < min_from_start:
                return None
        return gap[0]

    def update_gap_fallback(self, run):
        # Without Nav2: wall-follow until a real opening is seen, then turn in.
        nav = run.spec['nav2']
        side = nav.get('side', 'left')
        gap = self.fresh_gap(run, side, float(nav.get('min_ahead', -1.0)))
        if gap is not None:
            self.plan_turn_in(run, side, nav, gap)

    def plan_turn_in(self, run, side, nav, gap):
        """Turn into a side opening, planned in the opening's own frame (X through it,
        wall line X=0) so it doesn't depend on the current heading or wall distance:
        follow the line X = -(R + a + end_clear) along the wall, start a full-lock turn
        when the camera is R - a before the opening's axis (the pivot is the rear axle,
        a behind the camera), end square to it end_clear before the wall, then follow
        its axis in. Reverse first if the opening was seen too late to line up."""
        self.set_entrance(gap, side)
        gx, gy, gyaw = self.last_gap
        s = 1.0 if side == 'left' else -1.0
        radius = float(nav.get('turn_radius', self.variables['turn_radius']))
        a = self.variables['axle_to_camera']
        speed = float(nav.get('turn_speed', 0.4))
        end_clear = float(self.mission.get('gap_entry', {}).get('end_clear', 0.3))
        c, sn = math.cos(gyaw), math.sin(gyaw)
        x_line = -(radius + a + end_clear)
        y_turn = s * (radius - a)
        px, py = gx + c * x_line - sn * y_turn, gy + sn * x_line + c * y_turn
        dx, dy = self.odom.x - gx, self.odom.y - gy
        x0, y0 = c * dx + sn * dy, -sn * dx + c * dy
        approach = wrap_angle(gyaw - s * math.pi / 2)
        run_up = s * (y0 - y_turn)
        need = abs(x0 - x_line) / math.tan(math.radians(30.0)) + 0.3
        queue = []
        if run_up < need:
            queue.append(('backoff', {'speed': 0.3, 'distance': need - run_up}))
        queue.append(('heading_hold', {'speed': speed, 'heading_abs': approach, 'line_point': [px, py],
                                       'max_approach_deg': 30.0, 'past_point': 0.0}))
        queue.append(('arc_turn', {'direction': side, 'heading_abs': gyaw, 'speed': speed,
                                   'radius': radius, 'allow_kturn': True}))
        inside = float(nav.get('inside', 0.0))
        if inside > 0.0:
            queue.append(('heading_hold', {'speed': speed, 'heading_abs': gyaw,
                                           'line_point': [gx, gy], 'max_approach_deg': 30.0,
                                           'past_point': inside}))
        self.variables['gap_yaw'] = gyaw
        self.get_logger().info(f'{side} gap at ({gx:.2f}, {gy:.2f}): run-up {run_up:.2f} m '
                               f'(need {need:.2f}), off-line {x0 - x_line:+.2f} m -> '
                               f'{[q[0] for q in queue]}')
        self.nav.cancel()
        run.fallback = False
        run.queue = queue
        run.phase = 'turn_in'
        self.next_in_queue(run)

    def next_in_queue(self, run):
        if run.queue:
            name, params = run.queue.pop(0)
            self.send_behavior(name, params)
        else:
            run.done = True

    def update_nav(self, run):
        nav = run.spec['nav2']
        kind = nav.get('goal', 'relative')
        if kind == 'relative':
            if run.phase is None:
                # frame: gap anchors the goal to the opening we just entered, whose
                # direction comes from wall geometry rather than our skewed heading.
                base = run.pose0
                if nav.get('frame') == 'gap' and self.last_gap is not None:
                    base = self.last_gap
                self.nav.send(*self.relative(base, float(nav.get('forward', 1.0)),
                                             float(nav.get('left', 0.0)),
                                             float(nav.get('yaw_deg', 0.0))))
                run.phase = 'go'
            elif self.nav.state == 'succeeded':
                run.done = True
            elif self.nav.state == 'failed':
                self.nav_failed(run, 'relative goal')
            return

        if kind == 'gap':
            # Search by following the wall (works for any course length). Entry is
            # a deterministic turn-in by default; entry: nav2 lets Nav2 plan it.
            side = nav.get('side', 'left')
            if run.phase == 'plan':
                status = self.bstatus.get(self.cmd_id)
                if status == 'done':
                    run.done = True
                elif status == 'failed' and self.last_gap is not None:
                    self.get_logger().warn('entry plan failed; geometric turn-in')
                    self.plan_turn_in(run, side, nav, self.last_gap)
                return
            if run.phase == 'turn_in':
                if self.bstatus.get(self.cmd_id) in ('done', 'failed'):
                    self.next_in_queue(run)
                return
            if run.phase in (None, 'search'):
                mfs = nav.get('min_from_start')
                gap = self.fresh_gap(run, side, float(nav.get('min_ahead', -1.0)),
                                     None if mfs is None else float(mfs))
                if gap is not None:
                    gx, gy, gyaw = gap
                    self.get_logger().info(f'{side} gap found at ({gx:.2f}, {gy:.2f})')
                    entry = nav.get('entry', 'turn_in')
                    if entry == 'plan':
                        # Hybrid-A* on the lidar map: plans around decoys, reverses if needed.
                        self.set_entrance(gap, side)
                        gyaw = self.last_gap[2]
                        inside = float(nav.get('inside', 0.6))
                        self.send_behavior('plan_drive', {
                            'goal': [gx + inside * math.cos(gyaw), gy + inside * math.sin(gyaw), gyaw],
                            'speed': float(nav.get('turn_speed', 0.5))})
                        run.phase = 'plan'
                        return
                    if entry != 'nav2' or not self.nav_usable():
                        self.plan_turn_in(run, side, nav, gap)
                        return
                    inside = float(nav.get('inside', 0.8))
                    self.send_behavior('idle', {})
                    self.nav.send(gx + inside * math.cos(gyaw), gy + inside * math.sin(gyaw), gyaw)
                    run.phase = 'enter'
                    return
            if run.phase is None:
                search = run.spec.get('fallback') or {
                    'behavior': 'wall_center', 'params': {'follow': side, 'speed': 0.5}}
                self.send_behavior(search['behavior'], search.get('params', {}))
                run.phase = 'search'
            elif run.phase == 'enter':
                if self.nav.state == 'succeeded':
                    run.done = True
                elif self.nav.state == 'failed':
                    gap = self.gaps.get(side)
                    if gap:
                        self.plan_turn_in(run, side, nav, gap[0])
            return
        raise ValueError(f'unknown nav2 goal type {kind!r}')

    def reject(self, why):
        """A sensed gate not taken: say why (throttled), for tuning on the course."""
        now = self.now_s()
        if now - getattr(self, 'last_reject_log', -1e9) > 1.0:
            self.last_reject_log = now
            self.get_logger().info(f'gate rejected: {why}')
        return None

    def next_gate(self, run, spec):
        if self.gate is None:
            return None
        (gx, gy, gyaw), t = self.gate
        if t < run.t0 or self.now_s() - t > 1.0:
            return None
        fx, fy = self.odom.to_robot(gx, gy)
        if fx < 0.3 or math.hypot(fx, fy) > float(spec.get('max_range', 3.0)):
            return self.reject(f'range fx={fx:.2f} fy={fy:.2f}')
        if any(math.hypot(gx - px, gy - py) < 0.5 for px, py in self.passed_gates):
            return self.reject('already passed')
        # Hoops are passed roughly in the direction of travel.
        if abs(wrap_angle(gyaw - self.odom.yaw)) > math.radians(float(spec.get('max_angle_deg', 60))):
            return self.reject(f'angle {math.degrees(wrap_angle(gyaw - self.odom.yaw)):.0f}')
        # Known layout: each hoop stands on a fixed line of the lane (lane frame from the
        # bucket exit); anything else is a wall/bucket artefact.
        expect = spec.get('expect')
        if expect is not None and self.lane_origin is not None:
            ox, oy, oyaw = self.lane_origin
            # Hoops stand square to their lane: a pair whose axis disagrees is a mis-pairing
            # (one leg + a stray return, seen obliquely), whose centre is off as well.
            lay = self.square(wrap_angle(oyaw + math.radians(float(expect.get('yaw_deg', 0.0)))))
            if abs(wrap_angle(gyaw - lay)) > math.radians(float(expect.get('axis_tol_deg', 20.0))):
                return self.reject(f'axis {math.degrees(wrap_angle(gyaw - lay)):.0f} deg vs lane')
            # A hoop the depth image confirmed is taken as sensed: only its crossing
            # direction is checked, not where odometry says it should be.
            if self.gate_confirmed:
                return gx, gy, lay
            c, s = math.cos(oyaw), math.sin(oyaw)
            along, lat = c * (gx - ox) + s * (gy - oy), -s * (gx - ox) + c * (gy - oy)
            tol = float(expect.get('tol', 0.4))
            # Order, not distance: the next hoop lies beyond the last one passed (lane frame
            # is re-anchored on it). Odometry scale over a hoop spacing can be 30% off.
            prev = self.last_hoop_along
            ahead = along - prev
            if not 1.0 <= ahead <= float(expect.get('max_ahead', 4.5)):
                return self.reject(f'{ahead:.2f} m past the last hoop')
            if 'lat' in expect and abs(lat - float(expect['lat'])) > tol:
                return self.reject(f'lat {lat:.2f}')
            return gx, gy, lay
        # A hoop in a 72" lane can't be far off the lane line (rejects bucket/corner pairs).
        lane = run.pose0[2]
        off = -math.sin(lane) * (gx - self.odom.x) + math.cos(lane) * (gy - self.odom.y)
        if abs(off) > float(spec.get('max_offline', 1.2)):
            return None
        return gx, gy, gyaw

    def gate_lead_needed(self, lateral, radius, max_angle):
        """Run-up a car with turning radius R needs to shift sideways by `lateral`
        while capping its intercept angle at max_angle (S-curve + straight)."""
        e = abs(lateral)
        s_curve_shift = 2 * radius * (1 - math.cos(max_angle))
        if e <= s_curve_shift:
            theta = math.acos(max(-1.0, 1 - e / (2 * radius)))
            return 2 * radius * math.sin(theta)
        return 2 * radius * math.sin(max_angle) + (e - s_curve_shift) / math.tan(max_angle)

    def update_gate(self, run, now):
        """Hoop: follow the line through the gate centre along its axis; back up
        first when too close to make the sideways shift at our turning radius.
        No Nav2: reverse cusps in a 72\" lane were unreliable in simulation."""
        spec = run.spec['gate']
        beyond = float(spec.get('beyond', 0.6))
        speed = float(spec.get('commit_speed', 0.5))
        max_angle = math.radians(float(spec.get('max_approach_deg', 45.0)))
        radius = float(self.param('vehicle.planning_min_turn_radius', 1.52))
        lane_yaw = run.pose0[2]
        expect = spec.get('expect')
        if expect is not None and self.lane_origin is not None:
            # Known layout: search along the hoop's expected axis, not our current heading.
            lane_yaw = wrap_angle(self.lane_origin[2] + math.radians(float(expect.get('yaw_deg', 0.0))))

        if run.phase == 'approach':
            if self.bstatus.get(self.cmd_id) in ('done', 'failed'):
                if run.queue:
                    self.next_in_queue(run)
                else:
                    self.passed_gates.append(run.target[:2])
                    # The car wash steers blind on this through the ribbons: keep it on-grid.
                    gate_yaw = self.square(run.target[2])
                    if expect is not None and self.lane_origin is not None \
                            and abs(wrap_angle(gate_yaw - lane_yaw)) > math.radians(45):
                        # Close-range leg pairs can snap a quadrant off (R0.91 run sent
                        # the car wash 90 deg into a wall): the layout axis wins.
                        self.get_logger().warn(
                            f'gate axis {math.degrees(gate_yaw):.0f} deg disagrees with layout '
                            f'{math.degrees(lane_yaw):.0f} deg; using layout')
                        gate_yaw = self.square(lane_yaw)
                    self.variables['gate_yaw'] = gate_yaw
                    ex = spec.get('exit_goal')
                    if ex:
                        tx, ty = run.target[:2]
                        self.variables[ex.get('name', 'gate_exit')] = list(self.relative(
                            (tx, ty, gate_yaw), float(ex.get('forward', 0.0)),
                            float(ex.get('left', 0.0)), float(ex.get('yaw_deg', 0.0))))
                    self.reanchor_lane(spec.get('expect'), run.target)
                    run.done = True
                return
            # On the final line: re-aim at fresh sightings of the same hoop, so odometry
            # drift since it was first seen doesn't steer us into a leg.
            fresh = self.next_gate(run, spec) if not run.queue and 'target' not in spec else None
            if fresh is not None and run.target is not None and now - run.t_reaim > 0.5:
                moved = math.hypot(fresh[0] - run.target[0], fresh[1] - run.target[1])
                if 0.08 < moved < 0.8:
                    gx, gy, gyaw = fresh
                    run.target = fresh
                    run.t_reaim = now
                    self.send_behavior('heading_hold', {
                        'heading_abs': gyaw, 'line_point': [gx, gy], 'past_point': beyond,
                        'max_approach_deg': math.degrees(max_angle), 'speed': speed})
            return

        gate = self.next_gate(run, spec)
        if 'target' in spec:
            # Fixed gate (e.g. the bucket exit): known point, along its measured axis.
            tgt = self.resolve({'t': spec['target']}).get('t')
            axis = self.variables.get('exit_yaw', self.variables.get('gap_yaw', lane_yaw))
            gate = (tgt[0], tgt[1], axis) if tgt else None
        if gate is not None:
            gx, gy, gyaw = gate
            c, s = math.cos(gyaw), math.sin(gyaw)
            dx, dy = self.odom.x - gx, self.odom.y - gy
            before = -(c * dx + s * dy)
            lateral = -s * dx + c * dy
            need = self.gate_lead_needed(lateral, radius, max_angle) + float(spec.get('margin', 0.3))
            backup = need - before
            run.target = gate
            run.queue = []
            if backup > 0.05 and spec.get('plan'):
                # Not enough run-up: plan (forward/reverse) to a pose on the gate axis.
                lead = float(spec.get('plan_lead', 0.8))
                run.queue.append(('plan_drive', {
                    'goal': [gx - c * lead, gy - s * lead, gyaw], 'speed': speed, 'goal_tol': 0.2}))
            elif backup > 0.05:
                run.queue.append(('backoff', {'speed': 0.3, 'distance': backup}))
            run.queue.append(('heading_hold', {
                'heading_abs': gyaw, 'line_point': [gx, gy], 'past_point': beyond,
                'max_approach_deg': math.degrees(max_angle), 'speed': speed}))
            self.get_logger().info(
                f'gate at ({gx:.2f}, {gy:.2f}): {before:.2f} m ahead, {lateral:+.2f} m off-line, '
                f'needs {need:.2f} m run-up -> {"back up %.2f m" % backup if backup > 0.05 else "go"}')
            run.phase = 'approach'
            self.next_in_queue(run)
            return

        # No gate in view: creep along the lane direction (never the current heading).
        if run.phase == 'search' and self.bstatus.get(self.cmd_id, 'running') == 'running':
            return
        run.searches += 1
        if run.searches > int(spec.get('max_search', 4)):
            self.get_logger().warn('no gate found, moving on')
            # Later steps steer off $gate_yaw: use the hoop's expected axis instead.
            expect = spec.get('expect')
            if expect is not None and self.lane_origin is not None:
                self.variables['gate_yaw'] = wrap_angle(
                    self.lane_origin[2] + math.radians(float(expect.get('yaw_deg', 0.0))))
            run.failed = True
            return
        self.send_behavior('heading_hold', {'heading_abs': lane_yaw, 'speed': speed,
                                            'distance': float(spec.get('search_forward', 1.0))})
        run.phase = 'search'


def main(args=None):
    run_node(MissionManager, args)
