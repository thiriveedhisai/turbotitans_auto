"""Generate a Gazebo Harmonic world (SDF) from the same Course model the 2-D sim uses.

Static course: hay-bale walls, rails, ramp/bridge/helix/bank slabs, gravel and
pothole bumps, buckets, hoops, car-wash ribbons (visual only: seen by lidar and
camera, no collision), AprilTag markers on overhead panels, and the start light
(green disc behind a red cover the referee removes at race start).

Truck: Ackermann steering (rear drive), gpu_lidar, RGB-D camera, IMU, ground
truth odometry published as odom -> zed_camera_link. The model origin IS the
camera, so the odometry point matches the real ZED.
"""

import math
import os

import numpy as np

from tt_course_sim.course import Course

WHEEL_R = 0.065
CAM_H = 0.20                # camera height above ground
WHEELBASE = 0.32
AXLE_TO_CAMERA = 0.38
TRACK = 0.18


def _pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    return f'<pose>{x:.4f} {y:.4f} {z:.4f} {roll:.4f} {pitch:.4f} {yaw:.4f}</pose>'


def _box_link(name, pose, size, rgba, collide=True, extra_visual=''):
    sx, sy, sz = size
    col = (f'<collision name="c"><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>'
           f'</geometry><surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>'
           f'</surface></collision>') if collide else ''
    return (f'<link name="{name}">{pose}{col}<visual name="v"><geometry><box>'
            f'<size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>'
            f'<material><diffuse>{rgba}</diffuse><ambient>{rgba}</ambient>{extra_visual}'
            f'</material></visual></link>')


def _cyl_link(name, pose, r, h, rgba, collide=True):
    col = (f'<collision name="c"><geometry><cylinder><radius>{r}</radius><length>{h}</length>'
           f'</cylinder></geometry></collision>') if collide else ''
    return (f'<link name="{name}">{pose}{col}<visual name="v"><geometry><cylinder>'
            f'<radius>{r}</radius><length>{h}</length></cylinder></geometry>'
            f'<material><diffuse>{rgba}</diffuse><ambient>{rgba}</ambient></material>'
            f'</visual></link>')


def _course_links(c, tag_files, rng):
    links = []
    n = 0
    bale, rail = '0.85 0.7 0.3 1', '0.4 0.4 0.9 1'
    for x1, y1, b1, t1, x2, y2, b2, t2, kind in c.segs3d:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-3:
            continue
        thick = 0.12 if kind == 'bale' else 0.03
        h = ((t1 - b1) + (t2 - b2)) / 2
        pitch = -math.atan2(b2 - b1, length)
        cz = (b1 + b2) / 2 + h / 2
        links.append(_box_link(f'w{n}', _pose((x1 + x2) / 2, (y1 + y2) / 2, cz, 0, pitch,
                                              math.atan2(y2 - y1, x2 - x1)),
                               (length + thick * 0.5, thick, h), bale if kind == 'bale' else rail))
        n += 1
    for cx, cy, cz, length, width, thick, yaw, pitch, roll, zone in c.slabs3d:
        links.append(_box_link(f's{n}', _pose(cx, cy, cz, roll, pitch, yaw),
                               (length, width, thick), '0.6 0.6 0.6 1'))
        n += 1
    # Gravel / potholes: small bumps (holes can't be cut into a box).
    for cx, cy, yaw, length, width, z, vib in c.rough3d:
        count, h = (60, 0.012) if vib < 4 else (30, 0.019)
        cs, sn = math.cos(yaw), math.sin(yaw)
        for _ in range(count):
            u, v = rng.uniform(-length / 2, length / 2), rng.uniform(-width / 2 + 0.05, width / 2 - 0.05)
            links.append(_cyl_link(f'r{n}', _pose(cx + cs * u - sn * v, cy + sn * u + cs * v, z + h / 2),
                                   0.075, h, '0.45 0.4 0.35 1'))
            n += 1
    for x, y, r, kind in c.discs:
        if kind == 'bucket':
            links.append(_cyl_link(f'b{n}', _pose(x, y, 0.185), 0.15, 0.37, '1 0.5 0 1'))
        elif kind == 'hoop':
            links.append(_cyl_link(f'h{n}', _pose(x, y, 0.3), 0.025, 0.6, '0.8 0.1 0.1 1'))
        elif kind == 'post':
            links.append(_cyl_link(f'p{n}', _pose(x, y, 0.18), r, 0.36, '0.5 0.3 0.1 1'))
        n += 1
    # Hoop top bars (posts are listed in pairs).
    posts = [(x, y) for x, y, r, k in c.discs if k == 'hoop']
    for (x1, y1), (x2, y2) in zip(posts[0::2], posts[1::2]):
        links.append(_box_link(f'hb{n}', _pose((x1 + x2) / 2, (y1 + y2) / 2, 0.88, 0, 0,
                                               math.atan2(y2 - y1, x2 - x1)),
                               (math.hypot(x2 - x1, y2 - y1) + 0.05, 0.04, 0.04), '0.8 0.1 0.1 1'))
        n += 1
    # Car-wash ribbons: visual only, hanging to 0.1 m.
    for cx, cy, yaw, hl, hw in c.noise_boxes:
        cs, sn = math.cos(yaw), math.sin(yaw)
        for u in np.arange(-hl, hl, 0.4):
            for v in np.arange(-hw + 0.05, hw, 0.06):
                links.append(_box_link(f'rb{n}', _pose(cx + cs * u - sn * v, cy + sn * u + cs * v, 0.55),
                                       (0.004, 0.035, 0.9), '0.2 0.5 1 1', collide=False))
                n += 1
    # Zone markers: tag panels on the wall line just above the bales, facing the truck.
    for i, ((zone, x, y), yaw) in enumerate(zip(c.markers, c.marker_yaws)):
        if zone not in tag_files:
            continue
        mat = f'<pbr><metal><albedo_map>{tag_files[zone]}</albedo_map></metal></pbr>'
        links.append(_box_link(f'm{i}_{zone}', _pose(x, y, 0.50, 0, 0, yaw),
                               (0.005, 0.26, 0.26), '1 1 1 1', collide=False, extra_visual=mat))
    return '\n'.join(links)


def _start_light(c):
    x, y, yaw = c.start_pose
    lx, ly = x + 1.8 * math.cos(yaw), y + 1.8 * math.sin(yaw)
    face = yaw + math.pi
    green = (f'<model name="start_light"><static>true</static>'
             f'{_pose(lx, ly, 0.6, 0, math.pi / 2, face)}'
             f'{_cyl_link("disc", "<pose>0 0 0 0 0 0</pose>", 0.12, 0.01, "0 1 0 1", collide=False)}'
             f'</model>')
    cx, cy = lx - 0.03 * math.cos(yaw), ly - 0.03 * math.sin(yaw)
    cover = (f'<model name="start_cover"><static>true</static>'
             f'{_pose(cx, cy, 0.6, 0, math.pi / 2, face)}'
             f'{_cyl_link("disc", "<pose>0 0 0 0 0 0</pose>", 0.15, 0.01, "1 0 0 1", collide=False)}'
             f'</model>')
    return green + cover


def _wheel(name, x, y, z, steer=None):
    inertia = '<inertial><mass>0.15</mass><inertia><ixx>2e-4</ixx><iyy>3e-4</iyy><izz>2e-4</izz></inertia></inertial>'
    wheel = (f'<link name="{name}">{_pose(x, y, z, -math.pi / 2, 0, 0)}{inertia}'
             f'<collision name="c"><geometry><cylinder><radius>{WHEEL_R}</radius><length>0.05</length>'
             f'</cylinder></geometry><surface><friction><ode><mu>1.1</mu><mu2>1.1</mu2></ode></friction>'
             f'</surface></collision><visual name="v"><geometry><cylinder><radius>{WHEEL_R}</radius>'
             f'<length>0.05</length></cylinder></geometry><material><diffuse>0.1 0.1 0.1 1</diffuse>'
             f'</material></visual></link>')
    parent = steer or 'zed_camera_link'
    joint = (f'<joint name="{name}_joint" type="revolute"><parent>{parent}</parent><child>{name}</child>'
             f'<axis><xyz>0 0 1</xyz></axis></joint>')
    return wheel + joint


def _steer(name, x, y, z, limit):
    return (f'<link name="{name}">{_pose(x, y, z)}<inertial><mass>0.05</mass><inertia><ixx>1e-5</ixx>'
            f'<iyy>1e-5</iyy><izz>1e-5</izz></inertia></inertial></link>'
            f'<joint name="{name}_joint" type="revolute"><parent>zed_camera_link</parent>'
            f'<child>{name}</child><axis><xyz>0 0 1</xyz><limit><lower>{-limit:.4f}</lower>'
            f'<upper>{limit:.4f}</upper><effort>20</effort></limit></axis></joint>')


def truck_sdf(pose, radius):
    # Headroom: tyre slip makes the achieved radius ~10% wider than the geometric one,
    # ackermann_to_gz compensates with yaw_gain and needs the extra lock.
    limit = math.atan(WHEELBASE / (radius * 0.8))
    wz = WHEEL_R - CAM_H
    xf, xr = -AXLE_TO_CAMERA + WHEELBASE, -AXLE_TO_CAMERA
    half = TRACK / 2
    chassis = (
        '<link name="zed_camera_link">'
        '<inertial><pose>-0.2 0 -0.08 0 0 0</pose><mass>2.5</mass><inertia><ixx>0.02</ixx><iyy>0.06</iyy>'
        '<izz>0.07</izz></inertia></inertial>'
        '<collision name="c"><pose>-0.2 0 -0.07 0 0 0</pose><geometry><box><size>0.5 0.2 0.08</size></box>'
        '</geometry></collision>'
        '<visual name="v"><pose>-0.2 0 -0.07 0 0 0</pose><geometry><box><size>0.5 0.2 0.08</size></box>'
        '</geometry><material><diffuse>0.1 0.3 0.8 1</diffuse></material></visual>'
        '<sensor name="lidar" type="gpu_lidar"><pose>-0.15 0 0.05 0 0 0</pose><topic>/gz/scan</topic>'
        '<gz_frame_id>laser</gz_frame_id><update_rate>10</update_rate><always_on>1</always_on>'
        '<lidar><scan><horizontal><samples>360</samples><resolution>1</resolution>'
        '<min_angle>-3.14159</min_angle><max_angle>3.12414</max_angle></horizontal></scan>'
        '<range><min>0.15</min><max>8.0</max><resolution>0.01</resolution></range>'
        '<noise><type>gaussian</type><mean>0</mean><stddev>0.01</stddev></noise></lidar></sensor>'
        '<sensor name="zed" type="rgbd_camera"><topic>/gz/zed</topic><update_rate>15</update_rate>'
        '<always_on>1</always_on><gz_frame_id>zed_camera_link</gz_frame_id>'
        '<camera><horizontal_fov>1.6</horizontal_fov><image><width>640</width><height>360</height>'
        '<format>R8G8B8</format></image><clip><near>0.1</near><far>15</far></clip>'
        '<depth_camera><clip><near>0.3</near><far>10</far></clip></depth_camera></camera></sensor>'
        '<sensor name="imu" type="imu"><topic>/gz/imu</topic><update_rate>100</update_rate>'
        '<always_on>1</always_on><gz_frame_id>zed_imu_link</gz_frame_id></sensor>'
        '</link>')
    parts = [chassis,
             _steer('fl_steer', xf, half, wz, limit), _steer('fr_steer', xf, -half, wz, limit),
             _wheel('fl_wheel', xf, half, wz, 'fl_steer'), _wheel('fr_wheel', xf, -half, wz, 'fr_steer'),
             _wheel('rl_wheel', xr, half, wz), _wheel('rr_wheel', xr, -half, wz)]
    plugins = (
        '<plugin filename="gz-sim-ackermann-steering-system" name="gz::sim::systems::AckermannSteering">'
        '<left_joint>rl_wheel_joint</left_joint><right_joint>rr_wheel_joint</right_joint>'
        '<left_steering_joint>fl_steer_joint</left_steering_joint>'
        '<right_steering_joint>fr_steer_joint</right_steering_joint>'
        f'<kingpin_width>{TRACK}</kingpin_width><steering_limit>{limit:.4f}</steering_limit>'
        f'<wheel_base>{WHEELBASE}</wheel_base><wheel_separation>{TRACK}</wheel_separation>'
        f'<wheel_radius>{WHEEL_R}</wheel_radius><min_velocity>-1</min_velocity>'
        '<max_velocity>3</max_velocity><min_acceleration>-5</min_acceleration>'
        '<max_acceleration>3</max_acceleration><topic>/gz_cmd_vel</topic>'
        # Default P gain gave a ~1 s steering time constant (truck kept turning
        # after the helix); a hobby servo is ~0.15 s.
        '<steer_p_gain>10.0</steer_p_gain></plugin>'
        '<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">'
        '<odom_frame>odom</odom_frame><robot_base_frame>zed_camera_link</robot_base_frame>'
        '<odom_publish_frequency>30</odom_publish_frequency><odom_topic>/gz/odom</odom_topic>'
        '<tf_topic>/gz/tf</tf_topic><dimensions>3</dimensions></plugin>')
    x, y, z, yaw = pose
    return (f'<model name="truck">{_pose(x, y, z + CAM_H + 0.005, 0, 0, yaw)}'
            + ''.join(parts) + plugins + '</model>')


def make_tags(out_dir, zones_to_ids, size_px=400, border=0.15):
    """Render AprilTag 36h11 PNGs with a white border; returns zone -> file."""
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    gen = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
    files = {}
    pad = int(size_px * border)
    for zone, tag_id in zones_to_ids.items():
        img = gen(dictionary, int(tag_id), size_px)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        path = os.path.join(out_dir, f'tag_{zone}.png')
        cv2.imwrite(path, img)
        files[zone] = path
    return files


def generate(out_dir, seed=0, rails=True, radius=1.52, start_zone='', tag_map=None):
    """Write world.sdf + tag textures into out_dir. Returns (world_path, course, spawn)."""
    os.makedirs(out_dir, exist_ok=True)
    c = Course(rails=rails, seed=seed)
    rng = np.random.default_rng(seed + 100)
    tag_files = make_tags(out_dir, tag_map or {}) if tag_map else {}
    sx, sy, syaw = c.waypoint(start_zone) if start_zone else c.start_pose
    sz = c.ground(sx, sy, 0.0)[0]
    world = f'''<?xml version="1.0"?>
<sdf version="1.9">
<world name="course">
  <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
  <scene><ambient>0.7 0.7 0.7 1</ambient><background>0.8 0.85 0.9 1</background></scene>
  <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
    <diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.3 0.2 -0.9</direction></light>
  <model name="ground"><static>true</static>
    {_box_link("g", _pose(0, 0, -0.05), (80, 80, 0.1), "0.3 0.32 0.3 1")}
  </model>
  <model name="course_static"><static>true</static>
    {_course_links(c, tag_files, rng)}
  </model>
  {_start_light(c) if not start_zone else ''}
  {truck_sdf((sx, sy, sz, syaw), radius)}
</world>
</sdf>
'''
    path = os.path.join(out_dir, 'world.sdf')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(world)
    return path, c, (sx, sy, sz, syaw)


if __name__ == '__main__':
    import sys
    p, course, spawn = generate(sys.argv[1] if len(sys.argv) > 1 else '/tmp/tt_gz')
    print(p, 'links:', len(course.segs3d) + len(course.slabs3d), 'spawn', spawn)
