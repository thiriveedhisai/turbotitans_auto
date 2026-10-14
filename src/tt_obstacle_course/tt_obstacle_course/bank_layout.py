"""Bank turnaround layout from what the lidar sees at the lane mouth (pure, testable)."""

import math


def bank_layout(scan, pose, wall_end, cfg):
    """Fence and goals for the U-turn on the bank plane, in odometry coordinates.

    A driver judges the turnaround from the road they're on, not from where the car points:
    the entry lane's walls (RANSAC lines, so the flare and posts don't bend them) give the
    lane direction relative to the truck, and the plane sits at a known angle to that lane
    (`lane_angle_deg`, + = plane axis left of the lane). Global odometry heading error
    cancels: everything is laid out relative to the current pose.

    scan: ScanView at the trigger; pose: (x, y, yaw) of the odometry point (camera);
    wall_end: m ahead of the lidar where the lane walls end (= the plane's near edge).
    cfg (mission.yaml bank_entry), plane frame u = into the plane, v = left, origin = lane
    mouth centre on the near edge: plane_depth, north, south, return_offset (return lane
    centre at the edge), lane_angle_deg, margin, lidar_behind_camera.
    Returns dict: fence, enter_line, axis (odom yaw of u), goal, info."""
    x, y, yaw = pose
    lb = float(cfg.get('lidar_behind_camera', 0.15))
    lines = [scan.wall_line(s, max_dist=1.4, x_min=-2.0, x_max=wall_end) for s in (1, -1)]
    lines = [ln for ln in lines if ln is not None]
    lane = math.atan(sum(ln[1] for ln in lines) / len(lines)) if lines else 0.0
    if abs(lane) > math.radians(25):
        lane = 0.0
    # Lane mouth centre (lidar frame): mid-point of the two wall lines where they end.
    mid = (sum(ln[0] + ln[1] * wall_end for ln in lines) / len(lines)) if len(lines) == 2 else 0.0
    axis = yaw + lane - math.radians(float(cfg.get('lane_angle_deg', 0.0)))
    cl, sl = math.cos(yaw), math.sin(yaw)
    mx = x + cl * (wall_end - lb) - sl * mid
    my = y + sl * (wall_end - lb) + cl * mid
    c, s = math.cos(axis), math.sin(axis)

    def w(u, v):
        return [mx + c * u - s * v, my + s * u + c * v]
    depth = float(cfg.get('plane_depth', 1.219))
    north, south = float(cfg.get('north', 0.6)), float(cfg.get('south', -2.6))
    m = float(cfg.get('margin', 0.15))
    far, top, bot = depth - m, north - m, south + m
    fence = [w(far, top) + w(far, bot), w(0.0, top) + w(far, top), w(0.0, bot) + w(far, bot)]
    # Behind: lane-end walls and the divider post at the near edge. The lidar is blind on
    # the tilted plane, so reversing legs stop on this line, not on a scan.
    rear_fence = [w(-0.25, top) + w(-0.25, bot)]
    post = cfg.get('divider')
    if post:
        # The post between the two lane mouths: a square round it, for both directions.
        pu, pv, pr = (float(v) for v in post)
        sq = [w(pu - pr, pv - pr), w(pu + pr, pv - pr), w(pu + pr, pv + pr), w(pu - pr, pv + pr)]
        fence += [sq[i] + sq[(i + 1) % 4] for i in range(4)]
    ret = float(cfg.get('return_offset', -1.9))
    ra = math.radians(float(cfg.get('return_angle_deg', 0.0)))
    back = axis + math.pi + ra           # return lane heading (out of the plane)
    return {'fence': fence, 'rear_fence': rear_fence, 'axis': axis,
            'enter_line': w(0.0, top) + w(0.0, bot),
            'return_yaw': back,
            'goal': w(-0.6, ret - 0.6 * math.tan(ra)) + [back],
            'info': f'lane {math.degrees(lane):+.1f} deg, plane axis '
                    f'{math.degrees(axis - yaw):+.1f} deg off heading, mouth {mid:+.2f} m, '
                    f'{len(lines)} wall line(s)'}
