"""Mission manager: walks the course zone by zone from config/mission.yaml.

Each zone is a list of steps. A step is one of:
  behavior: <name>  params: {...}               -> behavior_executor drives
  nav2: {goal: relative|gap, ...}  fallback: {behavior, params}
  gate: {approach, beyond, search_forward, ...}  (hoops: Nav2 to a pre-gate
        pose, then a straight blind commit through)
A step ends when any 'until' trigger fires (default: done):
  done | failed | terrain=X | terrain!=X | marker=X | distance>=m | time>=s |
  yaw>=deg | gap=left|right | walls_lost>=m
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

from tt_obstacle_course.geometry import OdomTracker, quat_to_rpy, wrap_angle, yaw_to_quat
from tt_obstacle_course.ros_common import RELIABLE_SUB, TTNode, dumps, odom_update
from tt_obstacle_course.ros_common import run as run_node
from tt_obstacle_course.scan_utils import ScanView

TRIGGER = re.compile(r'^\s*(\w+)\s*(>=|!=|=)?\s*([\w.\-]+)?\s*$')
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
        self.pose0 = (odom.x, odom.y, odom.yaw)
        self.prev_yaw = odom.yaw
        self.turned = 0.0
        self.phase = None
        self.done = False
        self.failed = False
        self.fallback = False
        self.searches = 0
        self.target = None


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
        self.use_nav2 = bool(self.param('use_nav2', True))
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
        self.passed_gates = []
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
        for side in ('left', 'right'):
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
        now = self.now_s()
        self.markers[msg.data] = now
        j = self.zone_index.get(msg.data)
        if (self.started and not self.finished and j is not None and self.zone_idx >= 0
                and 0 < j - self.zone_idx <= self.skip_max
                and now - self.zone_t0 >= self.skip_min_time
                and self.zones[j].get('allow_marker_skip', True)):
            self.get_logger().warn(f'marker for {msg.data}: jumping from {self.zone["name"]}')
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

    # ------------------------------------------------------------ commands
    @property
    def zone(self):
        return self.zones[self.zone_idx]

    def send_behavior(self, name, params):
        self.cmd_id += 1
        self.cmd = {'id': self.cmd_id, 'behavior': name, 'params': params}
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

    def next_step(self):
        self.nav.cancel()
        self.step_idx += 1
        steps = self.zone.get('steps', [])
        if self.step_idx >= len(steps):
            self.enter_zone(self.zone_idx + 1)
            return
        spec = steps[self.step_idx]
        self.run = StepRun(spec, self.now_s(), self.odom)
        self.get_logger().info(f'step {self.step_idx}: {self.describe(spec)}')
        if 'behavior' in spec:
            self.send_behavior(spec['behavior'], spec.get('params', {}))
        elif 'nav2' in spec or 'gate' in spec:
            if not self.nav_usable() and 'gate' not in spec:
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
        if kind == 'marker':
            return self.markers.get(val, -1e9) >= run.t0
        if kind == 'distance':
            return self.odom.distance - run.d0 >= float(val)
        if kind == 'time':
            return now - run.t0 >= float(val)
        if kind == 'yaw':
            return abs(math.degrees(run.turned)) >= float(val)
        if kind == 'gap':
            g = self.gaps.get(val)
            return g is not None and g[1] >= run.t0
        if kind == 'walls_lost':
            if not self.lidar_ok or self.scan is None:
                return False
            return min(self.scan.side_distance(+1), self.scan.side_distance(-1)) >= float(val)
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
            self.enter_zone(0)
            return
        run = self.run
        run.turned += wrap_angle(self.odom.yaw - run.prev_yaw)
        run.prev_yaw = self.odom.yaw

        if now - self.zone_t0 > float(self.zone.get('timeout_s', 1e9)):
            self.get_logger().warn(f'zone {self.zone["name"]} timed out')
            self.enter_zone(self.zone_idx + 1)
            return
        if not run.fallback:
            if 'nav2' in run.spec:
                self.update_nav(run)
            elif 'gate' in run.spec:
                self.update_gate(run, now)
        elif run.spec.get('nav2', {}).get('goal') == 'gap':
            self.update_gap_fallback(run)
        why = self.step_finished(now)
        if why:
            self.get_logger().info(f'step {self.step_idx} ended: {why}')
            self.next_step()

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

    def fresh_gap(self, run, side, min_ahead):
        gap = self.gaps.get(side)
        if gap is None or gap[1] < run.t0 or self.now_s() - gap[1] > 1.0:
            return None
        gx, gy, _ = gap[0]
        return gap[0] if self.odom.to_robot(gx, gy)[0] > min_ahead else None

    def update_gap_fallback(self, run):
        # Without Nav2: wall-follow until a real opening is abeam, then turn in.
        if run.phase == 'fallback_turn':
            return
        nav = run.spec['nav2']
        side = nav.get('side', 'left')
        gap = self.fresh_gap(run, side, float(nav.get('fallback_turn_ahead', 0.3)))
        if gap is None:
            return
        self.get_logger().info(f'{side} gap found (fallback), turning in')
        self.send_behavior('arc_turn', {'direction': side, 'turn_deg': 90.0,
                                        'speed': float(nav.get('fallback_turn_speed', 0.4)),
                                        'allow_kturn': True})
        run.phase = 'fallback_turn'

    def update_nav(self, run):
        nav = run.spec['nav2']
        kind = nav.get('goal', 'relative')
        if kind == 'relative':
            if run.phase is None:
                self.nav.send(*self.relative(run.pose0, float(nav.get('forward', 1.0)),
                                             float(nav.get('left', 0.0)),
                                             float(nav.get('yaw_deg', 0.0))))
                run.phase = 'go'
            elif self.nav.state == 'succeeded':
                run.done = True
            elif self.nav.state == 'failed':
                self.nav_failed(run, 'relative goal')
            return

        if kind == 'gap':
            side = nav.get('side', 'left')
            if run.phase in (None, 'search'):
                gap = self.fresh_gap(run, side, float(nav.get('min_ahead', 0.3)))
                if gap is not None:
                    gx, gy, gyaw = gap
                    inside = float(nav.get('inside', 0.8))
                    self.nav.send(gx + inside * math.cos(gyaw), gy + inside * math.sin(gyaw), gyaw)
                    run.phase = 'enter'
                    self.get_logger().info(f'{side} gap found at ({gx:.2f}, {gy:.2f})')
                    return
            if run.phase is None:
                search = nav.get('search', {})
                self.nav.send(*self.relative(run.pose0, float(search.get('forward', 5.0)),
                                             float(search.get('left', 0.0))))
                run.phase = 'search'
            elif run.phase == 'search' and self.nav.state in ('succeeded', 'failed'):
                self.nav_failed(run, 'gap search (no opening seen)')
            elif run.phase == 'enter':
                if self.nav.state == 'succeeded':
                    run.done = True
                elif self.nav.state == 'failed':
                    self.nav_failed(run, 'gap entry')
            return
        raise ValueError(f'unknown nav2 goal type {kind!r}')

    def next_gate(self, run, spec):
        if self.gate is None:
            return None
        (gx, gy, gyaw), t = self.gate
        if t < run.t0 or self.now_s() - t > 1.0:
            return None
        fx, fy = self.odom.to_robot(gx, gy)
        if fx < 0.3 or math.hypot(fx, fy) > float(spec.get('max_range', 3.0)):
            return None
        if any(math.hypot(gx - px, gy - py) < 0.5 for px, py in self.passed_gates):
            return None
        return gx, gy, gyaw

    def update_gate(self, run, now):
        spec = run.spec['gate']
        approach = float(spec.get('approach', 1.0))
        beyond = float(spec.get('beyond', 0.6))
        commit_speed = float(spec.get('commit_speed', 0.5))
        nav_ok = self.nav_usable()

        if run.phase in (None, 'search'):
            gate = self.next_gate(run, spec)
            if gate is not None:
                gx, gy, gyaw = gate
                run.target = gate
                self.get_logger().info(f'gate at ({gx:.2f}, {gy:.2f})')
                if nav_ok:
                    self.nav.send(gx - approach * math.cos(gyaw), gy - approach * math.sin(gyaw), gyaw)
                    run.phase = 'approach'
                else:
                    heading = math.atan2(gy - self.odom.y, gx - self.odom.x)
                    dist = math.hypot(gx - self.odom.x, gy - self.odom.y)
                    self.commit(run, heading, dist + beyond, commit_speed)
                return
            if nav_ok:
                searching = self.nav.state in ('pending', 'active')
            else:
                searching = self.bstatus.get(self.cmd_id, 'running') == 'running'
            if not (run.phase == 'search' and searching):
                run.searches += 1
                if run.searches > int(spec.get('max_search', 4)):
                    self.get_logger().warn('no gate found, moving on')
                    run.failed = True
                    return
                fwd = float(spec.get('search_forward', 1.5))
                if nav_ok:
                    self.nav.send(*self.relative((self.odom.x, self.odom.y, self.odom.yaw), fwd, 0.0))
                else:
                    self.send_behavior('heading_hold', {'speed': commit_speed, 'distance': fwd})
                run.phase = 'search'
            return

        if run.phase == 'approach':
            if self.nav.state == 'succeeded':
                self.commit(run, run.target[2], approach + beyond, commit_speed)
            elif self.nav.state == 'failed':
                self.get_logger().warn('gate approach failed, searching again')
                run.phase = 'search'
                self.nav.cancel()
            return

        if run.phase == 'commit' and self.bstatus.get(self.cmd_id) in ('done', 'failed'):
            self.passed_gates.append(run.target[:2])
            run.done = True

    def commit(self, run, heading, distance, speed):
        # Last stretch through the hoop is driven straight; no replanning.
        self.nav.cancel()
        self.send_behavior('heading_hold', {'heading_abs': heading, 'distance': distance,
                                            'speed': speed})
        run.phase = 'commit'


def main(args=None):
    run_node(MissionManager, args)
