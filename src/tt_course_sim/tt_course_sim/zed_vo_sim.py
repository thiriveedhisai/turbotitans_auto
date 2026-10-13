"""Stand-in for the ZED's visual-inertial odometry, computed from the simulated camera.

Frame-to-frame RGB-D visual odometry (ORB features, depth back-projection, PnP RANSAC)
for translation and yaw, IMU for roll/pitch and for yaw while vision is lost. Like the
real ZED it drifts, and it loses tracking where the image has little texture or light
(tunnel, ribbons, a bare wall filling the view): position then freezes until tracking
returns. Optional forced blackouts per course zone stress-test the stack.

Publishes /zed/zed_node/odom (as the ZED wrapper does; the stack's odom_guard owns the TF).
Ground truth is used only to decide forced blackouts, never for the estimate.
"""

import math

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu

from tt_course_sim.course import Course

# Optical (x right, y down, z forward) -> body (x forward, y left, z up).
OPT2BODY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class ZedVoSim(Node):

    def __init__(self):
        super().__init__('zed_vo_sim')
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.min_inliers = int(p('min_inliers', 25))
        self.coast_s = float(p('coast_s', 0.4))                 # IMU-propagate this long before 'lost'
        self.depth_noise = float(p('depth_noise', 0.004))       # sigma = k * d^2 (stereo)
        self.blackout = [z for z in str(p('blackout_zones', '')).split(',') if z]
        seed = int(p('seed', 0))
        gap_b = float(p('gap_b', -1.0))
        self.course = Course(rails=bool(p('rails', True)), seed=seed,
                             gap_b=gap_b if gap_b >= 0 else None) if self.blackout else None
        self.spawn = [float(v) for v in p('spawn', [0.0, 0.0, 0.0, 0.0])]
        self.rng = np.random.default_rng(seed + 11)
        self.orb = cv2.ORB_create(nfeatures=800, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = None
        self.depth = {}
        self.prev = None                 # (keypoints xy, descriptors, depth image)
        self.pos = np.zeros(3)
        self.yaw = 0.0
        self.roll = self.pitch = 0.0
        self.yaw0 = None                 # IMU yaw at start (odom starts at 0 like the ZED)
        self.gyro_t = None
        self.img_t = None
        self.vel = 0.0
        self.vel_body = np.zeros(3)
        self.vo_frame = False            # last frame had a VO solution (else gyro drives yaw)
        self.fail_since = None
        self.tracking = True
        self.truth_zone = ''
        self.lost_since = None
        self.pub = self.create_publisher(Odometry, '/zed/zed_node/odom', 10)
        self.create_subscription(CameraInfo, '/zed/zed_node/rgb/color/rect/camera_info', self.on_info, 2)
        self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.on_depth, 5)
        self.create_subscription(Image, '/zed/zed_node/rgb/color/rect/image', self.on_image, 5)
        self.create_subscription(Imu, '/zed/zed_node/imu/data', self.on_imu, qos_profile_sensor_data)
        if self.course is not None:
            self.create_subscription(Odometry, '/sim/odom_truth', self.on_truth, 10)

    # ------------------------------------------------------------ inputs
    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_depth(self, msg):
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self.depth[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = d
        while len(self.depth) > 5:
            self.depth.pop(next(iter(self.depth)))

    def on_truth(self, msg):
        sx, sy, sz, syaw = self.spawn
        q = msg.pose.pose.position
        c, s = math.cos(syaw), math.sin(syaw)
        x, y, z = sx + c * q.x - s * q.y, sy + s * q.x + c * q.y, sz + q.z
        # The start line sits in the car-wash exit: only black out once under way.
        if math.hypot(q.x, q.y) < 3.0 and self.truth_zone == '':
            return
        self.truth_zone = self.course.ground(x, y, z)[4] or self.truth_zone

    def on_imu(self, msg):
        roll, pitch, yaw_abs = _rpy(msg.orientation)
        self.roll, self.pitch = roll, pitch
        if self.yaw0 is None:
            self.yaw0 = yaw_abs
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.gyro_t is not None:
            # Heading from the (biased, noisy) gyro, as in VIO: frame-to-frame visual
            # rotation is far noisier than a MEMS gyro over a 3-minute run.
            w = _rot(roll, pitch, self.yaw) @ np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                                        msg.angular_velocity.z])
            self.yaw += float(w[2]) * (t - self.gyro_t)
        self.gyro_t = t

    def on_image(self, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if self.K is None or key not in self.depth or self.yaw0 is None:
            return
        depth = self.depth.pop(key)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
        ok, step, dyaw, inliers = self.estimate(pts, des)
        forced = self.truth_zone in self.blackout
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = t - self.img_t if self.img_t is not None else 0.0
        self.img_t = t
        if ok and dt > 1e-3:
            # Motion model (what VIO's IMU fusion enforces): a car doesn't slide sideways,
            # jump vertically, or change speed by >1 m/s between frames.
            v = step / dt
            ok = (abs(v[1]) < 0.3 and abs(v[2]) < 0.4 and abs(v[0]) < 3.0
                  and abs(v[0] - self.vel_body[0]) < 1.0)
        good = ok and not forced
        if good:
            self.pos += _rot(self.roll, self.pitch, self.yaw) @ step
            if dt > 1e-3:
                self.vel_body = step / dt
            self.fail_since = None
        else:
            if self.fail_since is None:
                self.fail_since = t
            if t - self.fail_since <= self.coast_s and dt > 0.0:
                # Short visual gap: coast on the last velocity (VIO keeps going on the IMU).
                self.pos += _rot(self.roll, self.pitch, self.yaw) @ (self.vel_body * dt)
            else:
                self.vel_body = np.zeros(3)
        self.vo_frame = good
        self.vel = float(self.vel_body[0])
        tracking = self.fail_since is None or t - self.fail_since <= self.coast_s
        if tracking != self.tracking:
            self.get_logger().info(f'tracking {"OK" if tracking else "LOST"} ({inliers} inliers, '
                                   f'{len(pts)} features{", forced blackout " + self.truth_zone if forced else ""})')
        self.tracking = tracking
        self.prev = (pts, des, depth) if des is not None and len(pts) else None
        self.publish(msg.header.stamp)

    # ------------------------------------------------------------ VO
    def estimate(self, pts, des):
        """Camera motion since the previous frame in the body frame: (ok, step, dyaw, inliers)."""
        if self.prev is None or des is None or len(pts) < self.min_inliers:
            return False, None, 0.0, 0
        ppts, pdes, pdepth = self.prev
        if pdes is None or len(ppts) < self.min_inliers:
            return False, None, 0.0, 0
        pairs = self.bf.knnMatch(pdes, des, k=2)
        good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < self.min_inliers:
            return False, None, 0.0, len(good)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        obj, imgp = [], []
        h, w = pdepth.shape
        for m in good:
            u, v = ppts[m.queryIdx]
            d = float(pdepth[min(h - 1, int(v)), min(w - 1, int(u))])
            if not (0.3 < d < 8.0) or not math.isfinite(d):
                continue
            d += self.rng.normal(0.0, self.depth_noise * d * d)
            obj.append(((u - cx) * d / fx, (v - cy) * d / fy, d))
            imgp.append(pts[m.trainIdx])
        if len(obj) < self.min_inliers:
            return False, None, 0.0, len(obj)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(np.array(obj, np.float64), np.array(imgp, np.float64),
                                                 self.K, None, iterationsCount=100, reprojectionError=3.0,
                                                 confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
        n = 0 if inl is None else len(inl)
        if not ok or n < self.min_inliers:
            return False, None, 0.0, n
        R, _ = cv2.Rodrigues(rvec)
        # Points map prev -> cur; the camera moved by the inverse.
        Rc, tc = R.T, -R.T @ tvec.reshape(3)
        step = OPT2BODY @ tc
        Rb = OPT2BODY @ Rc @ OPT2BODY.T
        dyaw = math.atan2(Rb[1, 0], Rb[0, 0])
        if np.linalg.norm(step) > 0.4 or abs(dyaw) > 0.5:
            return False, None, 0.0, n
        return True, step, dyaw, n

    # ------------------------------------------------------------ output
    def publish(self, stamp):
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = 'odom'
        out.child_frame_id = 'zed_camera_link'
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = (float(v) for v in self.pos)
        cr, sr = math.cos(self.roll / 2), math.sin(self.roll / 2)
        cp, sp = math.cos(self.pitch / 2), math.sin(self.pitch / 2)
        cy, sy = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        o = out.pose.pose.orientation
        o.w = cr * cp * cy + sr * sp * sy
        o.x = sr * cp * cy - cr * sp * sy
        o.y = cr * sp * cy + sr * cp * sy
        o.z = cr * cp * sy - sr * sp * cy
        out.twist.twist.linear.x = self.vel
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ZedVoSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
