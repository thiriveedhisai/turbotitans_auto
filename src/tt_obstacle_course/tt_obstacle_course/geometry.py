"""Small math helpers shared by the obstacle-course nodes (no ROS imports)."""

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_rpy(x, y, z, w):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_to_quat(yaw):
    """Return (x, y, z, w) for a pure yaw rotation."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class SteeringModel:
    """Maps path curvature to the AckermannDrive steering command.

    The steering command is treated as normalised: +/- max_steer_cmd is full
    lock, which produces min_turn_radius. Positive = left (REP-103).
    """

    def __init__(self, min_turn_radius, max_steer_cmd):
        self.min_turn_radius = float(min_turn_radius)
        self.max_steer_cmd = float(max_steer_cmd)

    @property
    def max_curvature(self):
        return 1.0 / self.min_turn_radius

    def curvature_to_steer(self, kappa):
        return clamp(kappa * self.min_turn_radius, -1.0, 1.0) * self.max_steer_cmd

    def steer_to_curvature(self, steer):
        frac = clamp(steer / self.max_steer_cmd, -1.0, 1.0) if self.max_steer_cmd else 0.0
        return frac / self.min_turn_radius

    def full_lock(self, direction):
        """direction: +1 left, -1 right."""
        return math.copysign(self.max_steer_cmd, direction)


class OdomTracker:
    """Planar pose, travelled distance, forward speed and yaw rate from odometry.

    Velocities are differentiated from pose because the ZED odom twist is not
    guaranteed to be populated.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.x = self.y = self.z = self.yaw = 0.0
        self.roll = self.pitch = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.distance = 0.0
        self.stamp = None

    @property
    def valid(self):
        return self.stamp is not None

    def update(self, stamp, x, y, z, qx, qy, qz, qw):
        roll, pitch, yaw = quat_to_rpy(qx, qy, qz, qw)
        if self.stamp is not None:
            dt = stamp - self.stamp
            if dt <= 1e-4:
                return
            dx, dy, dz = x - self.x, y - self.y, z - self.z
            step = math.sqrt(dx * dx + dy * dy + dz * dz)
            fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
            v = math.copysign(step, fwd) / dt
            w = wrap_angle(yaw - self.yaw) / dt
            self.speed += self.alpha * (v - self.speed)
            self.yaw_rate += self.alpha * (w - self.yaw_rate)
            self.distance += step
        self.stamp = stamp
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def to_odom(self, fx, fy):
        """Robot-frame point (forward, left) -> odom-frame (x, y)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * fx - s * fy, self.y + s * fx + c * fy

    def to_robot(self, ox, oy):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = ox - self.x, oy - self.y
        return c * dx + s * dy, -s * dx + c * dy
