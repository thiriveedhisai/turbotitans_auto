"""Live debug dashboard in a browser: what the truck sees (camera, lidar) and what it thinks
(zone, step, maneuver, commands, perception, every log line as one uniform timeline).
Read-only unless started with --control: then a Drive panel takes over motion with the arrow
keys (through the speed governor: collision stops and e-stop still apply) while the whole
stack keeps running, and an E-STOP button latches /estop.

  ros2 run tt_obstacle_course tt_dash               everything  -> http://<truck-ip>:8080
  ros2 run tt_obstacle_course tt_dash --control     + Drive panel and E-STOP
  ros2 run tt_obstacle_course tt_dash --view cam    camera + AprilTag placement check
  ros2 run tt_obstacle_course tt_dash --view lidar  lidar + map
  ros2 run tt_obstacle_course tt_dash --view log    decision timeline only
  ros2 run tt_obstacle_course tt_dash --log ~/runs/0612_1405.log [--follow]   a saved log

Start it before, during or after anything: a full run, start_zone:=, a cut-down mission,
a hand-sent behavior, a bench test with motor:=false, the sim, or `ros2 bag play` of a
tt_record bag. It shows whatever is being published. Every event it sees is also saved to
~/runs/dash_<time>.jsonl (open with tt_log or tt_dash --log).
"""

import argparse
import itertools
import json
import math
import os
import secrets
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tt_obstacle_course.dash_page import PAGE
from tt_obstacle_course.log_story import LEVELS, Story, read_events

SNAP_PERIOD = 0.2
TOPIC_DECAY = 0.9


def load_tuning(path=None):
    import yaml
    if path is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            path = os.path.join(get_package_share_directory('tt_obstacle_course'),
                                'config', 'tuning.yaml')
        except Exception:
            path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tuning.yaml')
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return {}
    cfg = dict(data.get('/**', {}).get('ros__parameters', {}))
    cfg['marker_detector'] = data.get('marker_detector', {}).get('ros__parameters', {})
    cfg['odom_guard'] = data.get('odom_guard', {}).get('ros__parameters', {})
    return cfg


class Hub:
    """Everything the page shows; ROS callbacks write, HTTP threads read."""

    def __init__(self, save_path=None):
        self.cv = threading.Condition()
        self.events = deque(maxlen=20000)
        self.n = 0
        self.story = Story()
        self.state = {}
        self.scan = None
        self.pose = None
        self.trail = []
        self.percep = {}
        self.tags = []
        self.topics = {}
        self.jpeg = None
        self.jpeg_seq = 0
        self.clients = 0
        self.cam_clients = 0
        self.map = None
        self.map_ver = 0
        self.loc_pose = None
        self.teleop_pub = None
        self.estop_pub = None
        self.mapping = None
        self.odocheck = None
        self.save = open(save_path, 'a') if save_path else None
        self.save_path = save_path

    def event(self, ev, save=True):
        with self.cv:
            ms = self.state.get('mission')
            # Started mid-run: the zone/race-start lines went by before we subscribed.
            if isinstance(ms, dict) and ms.get('started') and ev.get('t') is not None:
                if self.story.zone == '-' and ms.get('zone'):
                    self.story.zone, self.story.step = ms['zone'], str(ms.get('step', '-'))
                if self.story.t0 is None:
                    self.story.t0 = ev['t'] - float(ms.get('elapsed') or 0.0)
            self.story.feed(ev)
            self.events.append(ev)
            self.n += 1
            if save and self.save:
                self.save.write(json.dumps(ev) + '\n')
                self.save.flush()
            self.cv.notify_all()

    def since(self, idx):
        first = self.n - len(self.events)
        idx = max(idx, first)
        return list(itertools.islice(self.events, idx - first, None)), self.n

    def set(self, key, value):
        with self.cv:
            self.state[key] = value

    def note(self, msg, lvl='INFO'):
        print(f'[{lvl}] [{time.time():.6f}] [tt_dash]: {msg}', flush=True)
        self.event({'t': time.time(), 'node': 'tt_dash', 'lvl': lvl, 'msg': msg})

    def set_map(self, occ, known, origin, res):
        """occ/known: bool grids, row 0 at origin y."""
        import base64
        import cv2
        import numpy as np
        img = np.zeros(occ.shape + (4,), np.uint8)
        img[known] = (225, 225, 225, 70)
        img[occ] = (255, 255, 255, 255)
        ok, buf = cv2.imencode('.png', img[::-1])
        if ok:
            with self.cv:
                self.map = {'type': 'map', 'png': base64.b64encode(buf.tobytes()).decode(),
                            'origin': [float(origin[0]), float(origin[1])], 'res': float(res),
                            'w': int(occ.shape[1]), 'h': int(occ.shape[0])}
                self.map_ver += 1

    def tick(self, topic):
        now = time.monotonic()
        t = self.topics.get(topic)
        if t is None:
            self.topics[topic] = {'last': now, 'dt': None}
            return
        dt = now - t['last']
        t['dt'] = dt if t['dt'] is None else TOPIC_DECAY * t['dt'] + (1 - TOPIC_DECAY) * dt
        t['last'] = now

    def snapshot(self, trail_from):
        now = time.monotonic()
        with self.cv:
            percep_local = []
            if self.pose is not None:
                x, y, yaw = self.pose
                c, s = math.cos(yaw), math.sin(yaw)
                off = self.state.get('_laser', (0.0, 0.0))
                for name, (px, py, pyaw, t) in self.percep.items():
                    dx, dy = px - x, py - y
                    percep_local.append({'name': name, 'x': c * dx + s * dy - off[0],
                                         'y': -s * dx + c * dy - off[1],
                                         'yaw': pyaw - yaw, 'age': time.time() - t})
            return {'type': 'snap',
                    'state': {k: v for k, v in self.state.items() if not k.startswith('_')},
                    'scan': self.scan, 'pose': self.pose, 'loc_pose': self.loc_pose,
                    'tags': self.tags,
                    'percep': self.percep, 'percep_local': percep_local,
                    'trail': self.trail[trail_from:],
                    'topics': {k: {'hz': 1.0 / t['dt'] if t['dt'] else 0.0,
                                   'age': now - t['last']} for k, t in self.topics.items()}
                    }, len(self.trail)


# ---------------------------------------------------------------- camera / tags
class TagCheck:
    """Same detection as marker_detector, but also reports what it rejects and why."""

    FAMILIES = {'tag36h11': 'DICT_APRILTAG_36h11', 'tag25h9': 'DICT_APRILTAG_25h9',
                'tag16h5': 'DICT_APRILTAG_16h5', 'aruco_4x4_50': 'DICT_4X4_50'}

    def __init__(self, m):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        d = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.FAMILIES.get(m.get('family', 'tag36h11'), 'DICT_APRILTAG_36h11')))
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detect = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers
        else:
            p = cv2.aruco.DetectorParameters_create()
            self.detect = lambda g: cv2.aruco.detectMarkers(g, d, parameters=p)
        self.size = float(m.get('tag_size', 0.20))
        self.max_range = float(m.get('max_range', 6.0))
        self.trigger = float(m.get('trigger_range', 1.5))
        self.min_side = float(m.get('min_side_px', 12.0))
        self.zone_of = dict(zip([int(i) for i in m.get('tag_ids', [])], m.get('tag_zones', [])))
        s = self.size / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def run(self, gray, K, D):
        cv2, np = self.cv2, self.np
        corners, ids, rejected = self.detect(gray)
        if ids is None:
            corners, ids, rejected = self.detect(self.clahe.apply(gray))
        out = []
        for c, tag in zip(corners, [] if ids is None else ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float32)
            side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
            zone = self.zone_of.get(int(tag))
            dist = bearing = normal = None
            if K is not None:
                ok, rvec, tvec = cv2.solvePnP(self.obj, pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    t = tvec.flatten()
                    dist = float(np.linalg.norm(t))
                    bearing = math.degrees(math.atan2(-t[0], t[2]))
                    n = cv2.Rodrigues(rvec)[0][:, 2]
                    normal = math.degrees(math.atan2(-n[0], n[2]))
            if zone is None:
                status = 'unknown id'
            elif np.linalg.norm(pts[0] - pts[1]) < self.min_side:
                status = 'too small'
            elif dist is None:
                status = 'no camera_info'
            elif dist > self.max_range:
                status = 'too far'
            elif dist <= self.trigger:
                status = 'TRIGGER'
            else:
                status = 'seen'
            out.append({'id': int(tag), 'zone': zone, 'dist': dist, 'bearing': bearing,
                        'normal': normal, 'side_px': side, 'status': status,
                        'corners': pts.tolist()})
        return out, [r.reshape(4, 2) for r in (rejected or [])]


def draw_overlay(cv2, img, tags, rejected, state, K, bright):
    import numpy as np
    h, w = img.shape[:2]
    sc = max(w / 640.0, 0.6)
    col = {'TRIGGER': (60, 255, 60), 'seen': (40, 230, 240)}
    for r in rejected[:30]:
        cv2.polylines(img, [np.asarray(r, dtype='int32')], True, (90, 90, 90), 1)
    for t in tags:
        c = col.get(t['status'], (60, 60, 255))
        pts = [tuple(int(v) for v in p) for p in t['corners']]
        cv2.polylines(img, [np.array(pts)], True, c, max(2, int(2 * sc)))
        d = '-' if t['dist'] is None else f'{t["dist"]:.2f}m'
        b = '' if t['bearing'] is None else f' {t["bearing"]:+.0f}deg'
        label = f'#{t["id"]} {t["zone"] or "?"} {d}{b} {t["status"]}'
        x, y = min(p[0] for p in pts), min(p[1] for p in pts) - 6
        cv2.putText(img, label, (x, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (0, 0, 0), 3)
        cv2.putText(img, label, (x, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, c, 1)
    if K is not None:
        cy, cx, fx = int(K[1, 2]), K[0, 2], K[0, 0]
        for x0 in range(0, w, 24):
            cv2.line(img, (x0, cy), (x0 + 12, cy), (200, 200, 200), 1)
        cv2.putText(img, 'camera height', (4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * sc,
                    (200, 200, 200), 1)
        if bright is not None and not math.isnan(bright):
            u = int(cx - fx * math.tan(bright))
            cv2.line(img, (u, 0), (u, h), (255, 255, 255), 2)
            cv2.putText(img, 'bright exit', (u + 4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc,
                        (255, 255, 255), 1)
    ms, bs = state.get('mission') or {}, state.get('bstatus') or {}
    cmd = state.get('cmd')
    hud = (f'{ms.get("zone") or "-"}/{ms.get("step", "-")}  {bs.get("behavior") or "-"} '
           f'{bs.get("status") or ""}  t={ms.get("elapsed", 0)}s')
    if cmd:
        hud += f'  v={cmd[0]:+.2f} steer={math.degrees(cmd[1]):+.0f}'
    if state.get('lidar_ok') is False:
        hud += '  LIDAR GATED'
    cv2.rectangle(img, (0, 0), (w, int(22 * sc)), (0, 0, 0), -1)
    cv2.putText(img, hud, (6, int(16 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (255, 255, 255), 1)


# ---------------------------------------------------------------- ROS side
def start_ros(hub, cfg, args, ros_args):
    import rclpy
    from ackermann_msgs.msg import AckermannDrive
    from geometry_msgs.msg import PoseStamped, Vector3
    from nav_msgs.msg import Odometry
    from rcl_interfaces.msg import Log
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
    from std_msgs.msg import Bool, Float32, String

    from tt_obstacle_course.scan_utils import ScanView

    rclpy.init(args=ros_args)
    node = Node('tt_dash')
    flat = [float(v) for v in cfg.get('scan_self_mask', [])]
    mask = [tuple(flat[i:i + 3]) for i in range(0, len(flat) - 2, 3)]
    yaw_off, flip = float(cfg.get('scan_yaw_offset', 0.0)), bool(cfg.get('scan_flip', False))
    hub.set('_laser', (float(cfg.get('laser_offset_x', -0.15)), float(cfg.get('laser_offset_y', 0.0))))

    def sub(typ, topic, cb, qos=10, key=None):
        def wrap(msg):
            hub.tick(key or topic)
            cb(msg)
        node.create_subscription(typ, topic, wrap, qos)

    def js(key):
        def cb(msg):
            try:
                hub.set(key, json.loads(msg.data))
            except ValueError:
                hub.set(key, msg.data)
        return cb

    def on_log(m):
        if m.name == 'tt_dash':
            return
        hub.event({'t': m.stamp.sec + m.stamp.nanosec * 1e-9, 'node': m.name.split('.')[-1],
                   'lvl': LEVELS.get(m.level, 'INFO'), 'msg': m.msg})

    def on_scan(m):
        view = ScanView.from_msg(m, yaw_off, flip, self_mask=mask)
        raw = ScanView.from_msg(m, yaw_off, flip)
        import numpy as np
        ok = np.isfinite(view.ranges)
        cut = np.isfinite(raw.ranges) & ~ok
        step = max(1, int(ok.sum() // 900))

        def pts(sel, a, r):
            return [[round(float(rr * math.cos(aa)), 3), round(float(rr * math.sin(aa)), 3)]
                    for aa, rr in zip(a[sel][::step], r[sel][::step])]
        with hub.cv:
            hub.scan = {'pts': pts(ok, view.angles, view.ranges),
                        'masked': pts(cut, raw.angles, raw.ranges)}

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with hub.cv:
            hub.pose = (p.x, p.y, yaw)
            zone = (hub.state.get('mission') or {}).get('zone') or '-'
            last = hub.trail[-1] if hub.trail else None
            if last is None or math.hypot(p.x - last[0], p.y - last[1]) > 0.05 or last[2] != zone:
                if len(hub.trail) < 40000:
                    hub.trail.append((round(p.x, 3), round(p.y, 3), zone))

    def on_percep(name):
        def cb(m):
            p, q = m.pose.position, m.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            with hub.cv:
                hub.percep[name] = (p.x, p.y, yaw, time.time())
        return cb

    def on_map(m):
        import numpy as np
        g = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        if hub.mapping is not None and hub.mapping.session is not None:
            return
        hub.set_map(g >= 50, g >= 0, (m.info.origin.position.x, m.info.origin.position.y),
                    m.info.resolution)

    def on_loc_pose(m):
        p, q = m.pose.position, m.pose.orientation
        with hub.cv:
            hub.loc_pose = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    odom_topic = cfg.get('odom_topic', '/odom/fused')
    sub(Log, '/rosout', on_log, QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE))
    sub(String, '/mission/state', js('mission'))
    sub(String, '/behavior/status', js('bstatus'))
    sub(String, '/behavior/cmd', js('bcmd'))
    sub(String, '/marker/detections', js('detections'))
    sub(String, '/terrain/state', lambda m: hub.set('terrain', m.data))
    sub(Bool, '/terrain/lidar_ok', lambda m: hub.set('lidar_ok', m.data))
    sub(Bool, '/odom/degraded', lambda m: hub.set('odom_degraded', m.data))
    sub(Bool, '/estop', lambda m: hub.set('estop', m.data))
    sub(String, '/estop/hw_status', js('hw_estop'))
    sub(Bool, '/race_start', lambda m: hub.set('race_start', m.data))
    sub(Float32, '/mission/speed_cap', lambda m: hub.set('cap_mission', m.data))
    sub(Float32, '/terrain/speed_cap', lambda m: hub.set('cap_terrain', m.data))
    sub(Float32, '/perception/bright_exit', lambda m: hub.set('bright', m.data))
    sub(Vector3, '/terrain/attitude', lambda m: hub.set('attitude', (m.x, m.y, m.z)))
    sub(AckermannDrive, '/ackermann_cmd', lambda m: hub.set('cmd', (m.speed, m.steering_angle)))
    sub(AckermannDrive, '/ackermann_cmd_raw',
        lambda m: hub.set('cmd_raw', (m.speed, m.steering_angle)))
    sub(LaserScan, cfg.get('scan_topic', '/scan'), on_scan, qos_profile_sensor_data, key='scan')
    sub(Odometry, odom_topic, on_odom, 10, key='odom')
    for name in ('left_gap', 'right_gap', 'front_gap', 'gate'):
        sub(PoseStamped, f'/perception/{name}', on_percep(name))
    if args.control:
        hub.teleop_pub = node.create_publisher(AckermannDrive, '/teleop_cmd', 10)
        hub.estop_pub = node.create_publisher(Bool, '/estop', 10)
        hub.max_steer = float(cfg.get('vehicle', {}).get('max_steer_cmd', 0.5))
        hub.mapping = MapControl(hub, cfg, args.maps_dir)
        mc = hub.mapping
        sub(Odometry, '/odom/local', mc.on_odom, 20, key='odom_local')
        sub(Imu, cfg.get('imu_topic', '/zed/zed_node/imu/data'), mc.on_imu, qos_profile_sensor_data,
            key='imu')
        sub(LaserScan, cfg.get('scan_topic', '/scan'), mc.on_scan, qos_profile_sensor_data, key='scan_map')
        sub(Bool, '/terrain/lidar_ok', mc.on_lidar_ok, key='lidar_ok_map')
        sub(String, '/mission/state', lambda m: mc.on_label('/mission/state', m), key='mission_lbl')
        sub(String, '/zone_event', lambda m: mc.on_label('/zone_event', m), key='zone_event')
        node.create_timer(2.0, mc.tick)
        hub.odocheck = OdoCheck(hub, cfg, args.maps_dir, args.check_speed)
        sub(Odometry, '/odom/local', hub.odocheck.on_odom, 20, key='odom_check')
        sub(Imu, cfg.get('imu_topic', '/zed/zed_node/imu/data'), hub.odocheck.on_imu,
            qos_profile_sensor_data, key='imu_check')
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy
    sub(OccupancyGrid, '/map', on_map, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                  durability=DurabilityPolicy.TRANSIENT_LOCAL))
    sub(PoseStamped, '/localization/pose', on_loc_pose)
    sub(String, '/localization/status', js('loc'))

    if args_view_wants_camera(args):
        cam = Camera(hub, cfg, args)
        cam.subscribe = lambda: (
            node.create_subscription(CameraInfo, args.camera_info, cam.on_info, qos_profile_sensor_data),
            node.create_subscription(Image, args.image, cam.on_image, qos_profile_sensor_data),
            node.create_subscription(String, '/marker/debug', cam.on_debug, 2))
        cam.unsubscribe = lambda subs: [node.destroy_subscription(s) for s in subs]
        threading.Thread(target=cam.loop, daemon=True).start()
    return rclpy, node


def args_view_wants_camera(args):
    return args.view in ('all', 'cam') and not args.no_camera


class Camera:
    def __init__(self, hub, cfg, args):
        import cv2
        from cv_bridge import CvBridge
        self.cv2, self.bridge = cv2, CvBridge()
        self.hub, self.fps, self.width = hub, args.fps, args.width
        self.tags = TagCheck(cfg.get('marker_detector', {}))
        self.K = self.D = None
        self.msg = None
        self.debug, self.debug_t = None, -1e9
        self.lock = threading.Lock()

    def on_debug(self, m):
        try:
            self.debug, self.debug_t = json.loads(m.data), time.monotonic()
        except ValueError:
            pass

    def detections(self, img):
        """The stack's own marker_detector result if it is running, else detect here."""
        import numpy as np
        d = self.debug
        if d is not None and time.monotonic() - self.debug_t < 0.5:
            return d['tags'], [np.array(r) for r in d.get('rejected', [])], 'marker_detector'
        gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        tags, rejected = self.tags.run(gray, self.K, self.D)
        return tags, rejected, 'tt_dash (marker_detector not running)'

    def on_info(self, m):
        import numpy as np
        self.K = np.array(m.k, dtype=float).reshape(3, 3)
        self.D = np.array(m.d, dtype=float) if len(m.d) else np.zeros(5)

    def on_image(self, m):
        self.hub.tick('image')
        with self.lock:
            self.msg = m

    def loop(self):
        cv2, hub = self.cv2, self.hub
        subs, idle_since = None, time.monotonic()
        while True:
            time.sleep(1.0 / self.fps)
            # Images cost CPU the stack's own camera nodes need: only subscribe while watched.
            if hub.cam_clients > 0:
                idle_since = time.monotonic()
                if subs is None:
                    subs = self.subscribe()
            elif subs is not None and time.monotonic() - idle_since > 3.0:
                self.unsubscribe(subs)
                subs = None
                hub.topics.pop('image', None)
            with self.lock:
                m, self.msg = self.msg, None
            if m is None or hub.cam_clients == 0:
                continue
            try:
                img = self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')
            except Exception as e:
                hub.set('camera_error', str(e))
                continue
            tags, rejected, source = self.detections(img)
            with hub.cv:
                state = dict(hub.state)
            draw_overlay(cv2, img, tags, rejected, state, self.K, state.get('bright'))
            hub.set('tag_source', source)
            if img.shape[1] > self.width:
                f = self.width / img.shape[1]
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with hub.cv:
                    hub.jpeg = buf.tobytes()
                    hub.jpeg_seq += 1
                    hub.tags = [{k: v for k, v in t.items() if k != 'corners'} for t in tags]
                    hub.cv.notify_all()


# ---------------------------------------------------------------- mapping from the page
def workspace_maps_dir():
    """<workspace>/maps, found from where the package is installed; ~/maps otherwise."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('tt_obstacle_course')
        parts = os.path.realpath(share).split(os.sep)
        if 'install' in parts:
            return os.sep.join(parts[:len(parts) - 1 - parts[::-1].index('install')] + ['maps'])
    except Exception:
        pass
    return os.path.expanduser('~/maps')


class MapControl:
    """Start/stop a mapping session from the dashboard; records a bag of it alongside."""

    def __init__(self, hub, cfg, maps_dir):
        self.hub, self.cfg, self.maps_dir = hub, cfg, maps_dir
        self.session = None
        self.rec = None
        self.lock = threading.Lock()

    def start(self, auto_stop=True):
        from tt_obstacle_course.map_tools import MapSession
        with self.lock:
            if self.session is not None and not self.session.done:
                return self.session.out
            out = os.path.join(self.maps_dir, time.strftime('map_%m%d_%H%M%S'))
            os.makedirs(out, exist_ok=True)
            self.session = MapSession(self.cfg, out, auto_stop=auto_stop, log=lambda s: self.hub.note(s))
        self.start_bag(out)
        self.hub.note(f'map: started, saving to {out}. Drive the course slowly; '
                      + ('it stops by itself back at the start line (or press Stop & save).'
                         if auto_stop else 'no auto-stop: press Stop & save when done.'))
        return out

    def start_bag(self, out):
        import shutil
        import subprocess
        from tt_obstacle_course.record import TOPICS
        if shutil.which('ros2') is None:
            return
        log = open(os.path.join(out, 'bag.log'), 'w')
        self.rec = subprocess.Popen(['ros2', 'bag', 'record', '-o', os.path.join(out, 'bag')] + TOPICS,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def stop_bag(self):
        import signal
        if self.rec is not None and self.rec.poll() is None:
            os.killpg(self.rec.pid, signal.SIGINT)
            try:
                self.rec.wait(10)
            except Exception:
                os.killpg(self.rec.pid, signal.SIGKILL)
        self.rec = None

    def stop(self):
        with self.lock:
            ses = self.session
            if ses is None:
                return None
            path = ses.stop() if not ses.done else ses.saved
        self.stop_bag()
        self.publish(ses)
        self.hub.note(f'map: saved {path}  (use it: map:={path})')
        return path

    def publish(self, ses):
        cm = ses.course_map()
        if cm is not None:
            self.hub.set_map(cm.occ, cm.known, cm.origin, cm.res)
        self.hub.set('mapping', ses.status())

    def tick(self):
        ses = self.session
        if ses is None:
            return
        if ses.done and self.rec is not None:
            self.stop()
            return
        if not ses.done:
            self.publish(ses)

    def on_imu(self, m):
        if self.session is not None and not self.session.done:
            self.session.on_imu(m)

    def on_odom(self, m):
        if self.session is not None and not self.session.done:
            self.session.on_odom(m)

    def on_scan(self, m):
        with self.lock:
            if self.session is not None and not self.session.done:
                self.session.on_scan(m)

    def on_lidar_ok(self, m):
        if self.session is not None:
            self.session.gate.lidar_ok = m.data

    def on_label(self, topic, m):
        if self.session is not None:
            self.session.lab.feed(topic, m)


# ---------------------------------------------------------------- odometry check
class OdoCheck:
    """Measure odometry against two taped legs: start/end of leg 1, turn, start/end of leg 2.
    Gives the distance scale (true / odometry) and the heading error of the ZED and the gyro."""

    STEPS = ['put the truck at the START of tape 1, lined up with it, then Mark',
             'drive slowly to the END of tape 1 (straight, along it), stop, then Mark',
             'turn onto tape 2 (any way, K-turn is fine), line up at its START, then Mark',
             'drive slowly to the END of tape 2, stop, then Mark']

    def __init__(self, hub, cfg, out_dir, max_speed):
        self.hub, self.out_dir, self.max_speed = hub, out_dir, max_speed
        self.cur_scale = float(cfg.get('odom_guard', {}).get('odom_scale', 1.0))
        self.active = False
        self.marks = []
        self.odom = None
        self.gyro = 0.0
        self.imu_t = None
        self.true_m = (5.0, 5.0)
        self.true_deg = 90.0

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.odom = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def on_imu(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.imu_t is not None and 0.0 < t - self.imu_t < 0.5:
            self.gyro += m.angular_velocity.z * (t - self.imu_t)
        self.imu_t = t

    def status(self, extra=None):
        st = {'active': self.active, 'marks': len(self.marks), 'max_speed': self.max_speed,
              'next': self.STEPS[len(self.marks)] if self.active and len(self.marks) < 4 else None}
        if extra:
            st.update(extra)
        self.hub.set('odocheck', st)
        return st

    def start(self, d):
        self.true_m = (float(d.get('leg1_m', 5.0)), float(d.get('leg2_m', 5.0)))
        self.true_deg = float(d.get('turn_deg', 90.0))
        self.active, self.marks = True, []
        self.hub.note(f'odometry check: started (legs {self.true_m[0]:.2f} / {self.true_m[1]:.2f} m, '
                      f'turn {self.true_deg:.0f} deg, teleop capped at {self.max_speed:.2f} m/s)')
        return self.status()

    def cancel(self):
        self.active, self.marks = False, []
        self.hub.note('odometry check: cancelled')
        return self.status()

    def mark(self):
        if not self.active:
            return self.status()
        if self.odom is None:
            return self.status({'error': 'no /odom/local yet'})
        self.marks.append((*self.odom, self.gyro))
        self.hub.note(f'odometry check: mark {len(self.marks)}/4 at x={self.odom[0]:.2f} y={self.odom[1]:.2f}')
        if len(self.marks) < 4:
            return self.status()
        self.active = False
        return self.status({'result': self.result()})

    def result(self):
        def wrapd(a):
            return math.degrees(math.atan2(math.sin(a), math.cos(a)))
        (x0, y0, v0, g0), (x1, y1, v1, g1), (x2, y2, v2, g2), (x3, y3, v3, g3) = self.marks
        d1, d2 = math.hypot(x1 - x0, y1 - y0), math.hypot(x3 - x2, y3 - y2)
        # NaN would break the page's JSON: a leg too short to measure is reported as None.
        s1 = self.true_m[0] / d1 if d1 > 0.5 else None
        s2 = self.true_m[1] / d2 if d2 > 0.5 else None
        path_turn = wrapd(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y0, x1 - x0))
        true_turn = math.copysign(self.true_deg, path_turn)
        zed_turn, gyro_turn = wrapd(v2 - v0), math.degrees(g2 - g0)
        good = [s for s in (s1, s2) if s is not None]
        scale = sum(good) / len(good) if good else None
        r = {'leg1_odom_m': round(d1, 3), 'leg2_odom_m': round(d2, 3),
             'scale_leg1': s1 and round(s1, 4), 'scale_leg2': s2 and round(s2, 4),
             'scale': scale and round(scale, 4),
             'suggested_odom_scale': scale and round(self.cur_scale * scale, 4),
             'warning': None if len(good) == 2 else 'a leg was under 0.5 m by odometry: redo the check',
             'true_turn_deg': true_turn,
             'path_turn_err_deg': round(path_turn - true_turn, 2),
             'zed_turn_err_deg': round(wrapd(math.radians(zed_turn - true_turn)), 2),
             'gyro_turn_err_deg': round(gyro_turn - true_turn, 2),
             'zed_drift_leg1_deg': round(wrapd(v1 - v0), 2), 'zed_drift_leg2_deg': round(wrapd(v3 - v2), 2),
             'gyro_drift_leg1_deg': round(math.degrees(g1 - g0), 2),
             'gyro_drift_leg2_deg': round(math.degrees(g3 - g2), 2)}
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, time.strftime('odocheck_%m%d_%H%M%S.json'))
        with open(path, 'w') as f:
            json.dump(dict(r, marks=self.marks, true_m=self.true_m, odom_scale_in_use=self.cur_scale), f, indent=1)
        r['saved'] = path
        sc = 'not measured (leg under 0.5 m)' if scale is None else \
            f'{scale:.3f} (set odom_guard odom_scale: {self.cur_scale * scale:.3f})'
        self.hub.note(f'odometry check: legs {d1:.2f} / {d2:.2f} m by odometry -> scale {sc}; '
                      f'turn error: path {r["path_turn_err_deg"]:+.1f}, ZED {r["zed_turn_err_deg"]:+.1f}, '
                      f'gyro {r["gyro_turn_err_deg"]:+.1f} deg; saved {path}')
        return r


# ---------------------------------------------------------------- HTTP side
def make_handler(hub, cfg, view, offline, token=None):
    hello = {'type': 'hello', 'cfg': {
        'view': view, 'offline': offline, 'vehicle': cfg.get('vehicle', {}),
        'control': token is not None,
        'laser_offset': [float(cfg.get('laser_offset_x', -0.15)),
                         float(cfg.get('laser_offset_y', 0.0))],
        'trigger_range': cfg.get('marker_detector', {}).get('trigger_range', 1.5)}}
    page = PAGE.replace('<body>', f'<body data-view="{view}" data-token="{token or ""}">').encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def do_POST(self):
            path = self.path.split('?', 1)[0]
            n = min(int(self.headers.get('Content-Length') or 0), 1024)
            body = self.rfile.read(n)
            if token is not None and hub.teleop_pub is None:
                return self.send(503, 'text/plain', b'starting up, try again in a moment')
            if token is None or hub.teleop_pub is None or path not in ('/teleop', '/estop', '/map/start',
                                                                        '/map/stop', '/odocheck'):
                return self.send(404, 'text/plain', b'control disabled (start tt_dash with --control)')
            # Random per-session token embedded in the page: other sites can't drive the truck.
            if self.headers.get('X-TT-Token') != token:
                return self.send(403, 'text/plain', b'bad token')
            try:
                d = json.loads(body or b'{}')
            except ValueError:
                return self.send(400, 'text/plain', b'bad json')
            from ackermann_msgs.msg import AckermannDrive
            from std_msgs.msg import Bool
            if path == '/estop':
                hub.estop_pub.publish(Bool(data=bool(d.get('on', True))))
            elif path == '/map/start':
                return self.send(200, 'text/plain',
                                 hub.mapping.start(bool(d.get('auto_stop', True))).encode())
            elif path == '/map/stop':
                return self.send(200, 'text/plain', (hub.mapping.stop() or '').encode())
            elif path == '/odocheck':
                oc, act = hub.odocheck, d.get('action')
                st = oc.start(d) if act == 'start' else oc.mark() if act == 'mark' else oc.cancel()
                return self.send(200, 'application/json', json.dumps(st).encode())
            else:
                v = max(-1.0, min(1.0, float(d.get('v', 0.0))))
                if hub.odocheck is not None and hub.odocheck.active:
                    v = max(-hub.odocheck.max_speed, min(hub.odocheck.max_speed, v))
                s = max(-hub.max_steer, min(hub.max_steer, float(d.get('steer', 0.0))))
                hub.teleop_pub.publish(AckermannDrive(speed=v, steering_angle=s))
            self.send(200, 'text/plain', b'ok')

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path in ('/', '/index.html'):
                self.send(200, 'text/html; charset=utf-8', page)
            elif path == '/events':
                self.events()
            elif path == '/stream.mjpg':
                self.mjpeg()
            elif path == '/frame.jpg':
                with hub.cv:
                    jpg = hub.jpeg
                if jpg:
                    self.send(200, 'image/jpeg', jpg)
                else:
                    self.send(404, 'text/plain', b'no camera frame yet')
            else:
                self.send(404, 'text/plain', b'not found')

        def send(self, code, ctype, body):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def stream_headers(self, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True

        def events(self):
            self.stream_headers('text/event-stream')
            w = self.wfile

            def put(obj):
                w.write(b'data: ' + json.dumps(obj, separators=(',', ':')).encode() + b'\n\n')
                w.flush()
            with hub.cv:
                hub.clients += 1
            try:
                put(hello)
                idx, trail_idx, last_snap, map_ver = 0, 0, 0.0, 0
                while True:
                    with hub.cv:
                        items, idx = hub.since(idx)
                        if not items:
                            hub.cv.wait(SNAP_PERIOD)
                            items, idx = hub.since(idx)
                        new_map = hub.map if hub.map_ver != map_ver else None
                        map_ver = hub.map_ver
                    if new_map:
                        put(new_map)
                    for i in range(0, len(items), 500):
                        put({'type': 'events', 'items': items[i:i + 500]})
                    if time.monotonic() - last_snap >= SNAP_PERIOD:
                        snap, trail_idx = hub.snapshot(trail_idx)
                        put(snap)
                        last_snap = time.monotonic()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with hub.cv:
                    hub.clients -= 1

        def mjpeg(self):
            self.stream_headers('multipart/x-mixed-replace; boundary=frame')
            with hub.cv:
                hub.cam_clients += 1
            seq = -1
            try:
                while True:
                    with hub.cv:
                        if hub.jpeg_seq == seq:
                            hub.cv.wait(1.0)
                        jpg, new = hub.jpeg, hub.jpeg_seq
                    if jpg is None or new == seq:
                        continue
                    seq = new
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                     + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with hub.cv:
                    hub.cam_clients -= 1
    return Handler


def feed_log(hub, path, follow):
    stream = sys.stdin if path == '-' else open(path, errors='replace')
    for ev in read_events(stream, follow):
        hub.event(ev, save=False)


def parse(argv, view_default='all'):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    ap.add_argument('--view', default=view_default, choices=['all', 'cam', 'lidar', 'map', 'log'])
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', default='0.0.0.0', help='127.0.0.1 = this machine only')
    ap.add_argument('--tuning', help='tuning.yaml (default: the installed one)')
    ap.add_argument('--log', help="preload a launch log / .jsonl ('-' = stdin)")
    ap.add_argument('--follow', action='store_true', help='keep reading --log as it grows')
    ap.add_argument('--no-ros', action='store_true', help='only show --log, no live topics')
    ap.add_argument('--no-camera', action='store_true', help="don't subscribe to the image")
    ap.add_argument('--fps', type=float, default=5.0, help='camera stream rate')
    ap.add_argument('--width', type=int, default=640, help='camera stream max width (px)')
    ap.add_argument('--image', default='/zed/zed_node/rgb/color/rect/image')
    ap.add_argument('--camera-info', default='/zed/zed_node/rgb/color/rect/camera_info')
    ap.add_argument('--save-dir', default=os.path.expanduser('~/runs'),
                    help="where the session .jsonl goes ('' = don't save)")
    ap.add_argument('--control', action='store_true',
                    help='enable the Drive panel (arrow keys -> /teleop_cmd), E-STOP and map making')
    ap.add_argument('--maps-dir', default='', help='where new maps go (default <workspace>/maps)')
    ap.add_argument('--check-speed', type=float, default=0.3,
                    help='teleop speed cap (m/s) while the odometry check is running')
    a = ap.parse_args(argv)
    a.maps_dir = os.path.expanduser(a.maps_dir) if a.maps_dir else workspace_maps_dir()
    return a


def main(argv=None, view_default='all'):
    raw = sys.argv[1:] if argv is None else argv
    ros_args = []
    if '--ros-args' in raw:
        i = raw.index('--ros-args')
        raw, ros_args = raw[:i], raw[i:]
    args = parse(raw, view_default)
    cfg = load_tuning(args.tuning)
    use_ros = not args.no_ros
    if use_ros:
        try:
            import rclpy  # noqa: F401
        except ImportError:
            print('rclpy not found: showing --log only')
            use_ros = False
    save = None
    if use_ros and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save = os.path.join(args.save_dir, time.strftime('dash_%m%d_%H%M%S.jsonl'))
    hub = Hub(save)
    if args.log:
        threading.Thread(target=feed_log, args=(hub, args.log, args.follow), daemon=True).start()

    token = secrets.token_hex(16) if args.control and use_ros else None
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(hub, cfg, args.view, not use_ros, token))
    server.daemon_threads = True
    host = 'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host
    print(f'tt_dash: open http://{host}:{args.port}/  (from another machine: the truck\'s IP)'
          + (f'\n  saving events to {save}' if save else ''), flush=True)
    if not use_ros:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ros, node = start_ros(hub, cfg, args, [sys.argv[0]] + ros_args)
    try:
        ros.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hub.mapping is not None and hub.mapping.session is not None and not hub.mapping.session.done:
            hub.mapping.stop()
        server.shutdown()
        node.destroy_node()
        if ros.ok():
            ros.shutdown()


def main_cam():
    main(view_default='cam')


def main_lidar():
    main(view_default='lidar')


if __name__ == '__main__':
    main()
