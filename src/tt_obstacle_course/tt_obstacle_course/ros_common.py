"""Shared ROS helpers for the obstacle-course nodes."""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from tt_obstacle_course.geometry import SteeringModel
from tt_obstacle_course.scan_utils import ScanView

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
# Volatile subscriber: matches both latched publishers and `ros2 topic pub`.
RELIABLE_SUB = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE)


class TTNode(Node):
    """Node that accepts any parameter from the tuning YAML without type fights."""

    def __init__(self, name):
        super().__init__(name, automatically_declare_parameters_from_overrides=True)

    def param(self, name, default):
        if self.has_parameter(name):
            value = self.get_parameter(name).value
            return default if value is None else value
        return self.declare_parameter(name, default).value

    def group(self, prefix):
        return {k: p.value for k, p in self.get_parameters_by_prefix(prefix).items()}

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def steering_model(self):
        return SteeringModel(float(self.param('vehicle.planning_min_turn_radius', 1.52)),
                             float(self.param('vehicle.max_steer_cmd', 0.5)))

    def scan_args(self):
        return (float(self.param('scan_yaw_offset', 0.0)), bool(self.param('scan_flip', False)))

    def scan_view(self, msg):
        """ScanView in the vehicle frame with our own parts (scan_self_mask) removed."""
        if not hasattr(self, '_scan_cfg'):
            flat = [float(v) for v in self.param('scan_self_mask', [0.0])]
            mask = [tuple(flat[i:i + 3]) for i in range(0, len(flat) - 2, 3)]
            self._scan_cfg = self.scan_args() + (mask,)
        yaw, flip, mask = self._scan_cfg
        return ScanView.from_msg(msg, yaw, flip, self_mask=mask)


def odom_update(tracker, msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    tracker.update(stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def dumps(obj):
    return json.dumps(obj, separators=(',', ':'))


def run(node_cls, args=None):
    rclpy.init(args=args)
    node = node_cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'shutdown'):
            node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
