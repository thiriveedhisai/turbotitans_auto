"""Gazebo <-> stack adapter for the 3-D sim.

- /ackermann_cmd (AckermannDrive) -> /gz_cmd_vel Twist for AckermannSteering. Stands in for
  rc_car_controller + PCA9685, with the same normalised steering model as the stack:
  +/-max_steer_cmd = full lock = min radius.
- /gz/odom_world (world-frame ground truth) -> /sim/odom_truth (referee, blackout zones), re-zeroed
  on the first message so it starts at 0 like the real ZED. With publish_odom (no simulated VO)
  it is also published as /zed/zed_node/odom (perfect odometry).
"""

import math

import rclpy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class AckermannToGz(Node):

    def __init__(self):
        super().__init__('ackermann_to_gz')
        self.radius = float(self.declare_parameter('turn_radius', 1.52).value)
        self.max_steer = float(self.declare_parameter('max_steer_cmd', 0.5).value)
        self.timeout = float(self.declare_parameter('timeout', 0.5).value)
        # Measured with steer_p_gain 10: full lock at R=1.07 gave 1.14 m, half lock 2.17 vs
        # 2.14 (tyre slip grows with lock); gain 1 + k*frac^2 matches turn_radius.
        self.yaw_gain = float(self.declare_parameter('yaw_gain_k', 0.065).value)
        self.publish_odom = bool(self.declare_parameter('publish_odom', False).value)
        self.cmd = AckermannDrive()
        self.last = None
        self.origin = None
        self.pub = self.create_publisher(Twist, '/gz_cmd_vel', 10)
        self.pub_truth = self.create_publisher(Odometry, '/sim/odom_truth', 10)
        self.pub_odom = self.create_publisher(Odometry, '/zed/zed_node/odom', 10) if self.publish_odom else None
        self.create_subscription(AckermannDrive, '/ackermann_cmd', self.on_cmd, 10)
        self.create_subscription(Odometry, '/gz/odom_world', self.on_odom, 10)
        self.create_timer(0.05, self.tick)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        if self.origin is None:
            self.origin = (p.x, p.y, p.z, _yaw(q))
        x0, y0, z0, yaw0 = self.origin
        c, s = math.cos(-yaw0), math.sin(-yaw0)
        dx, dy = p.x - x0, p.y - y0
        # q_rel = q_z(-yaw0) * q
        hw, hz = math.cos(-yaw0 / 2), math.sin(-yaw0 / 2)
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'odom'
        out.child_frame_id = 'zed_camera_link'
        out.pose.pose.position.x = c * dx - s * dy
        out.pose.pose.position.y = s * dx + c * dy
        out.pose.pose.position.z = p.z - z0
        o = out.pose.pose.orientation
        o.w = hw * q.w - hz * q.z
        o.x = hw * q.x - hz * q.y
        o.y = hw * q.y + hz * q.x
        o.z = hw * q.z + hz * q.w
        out.twist = msg.twist
        self.pub_truth.publish(out)
        if self.publish_odom:
            self.pub_odom.publish(out)

    def on_cmd(self, msg):
        self.cmd = msg
        self.last = self.get_clock().now()

    def tick(self):
        out = Twist()
        if self.last is not None and (self.get_clock().now() - self.last).nanoseconds / 1e9 < self.timeout:
            frac = max(-1.0, min(1.0, self.cmd.steering_angle / self.max_steer))
            out.linear.x = float(self.cmd.speed)
            out.angular.z = (1 + self.yaw_gain * frac * frac) * float(self.cmd.speed) * frac / self.radius
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannToGz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
