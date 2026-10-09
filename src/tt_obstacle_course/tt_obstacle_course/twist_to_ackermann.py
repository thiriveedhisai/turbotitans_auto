"""Nav2 Twist (/cmd_vel) -> AckermannDrive (/ackermann_cmd_raw).

Curvature = wz / vx, mapped through the shared SteeringModel so Nav2 and the
hand-written behaviors produce identical steering for the same curvature.
"""

from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist

from tt_obstacle_course.ros_common import TTNode, run


class TwistToAckermann(TTNode):

    def __init__(self):
        super().__init__('twist_to_ackermann')
        self.steering = self.steering_model()
        self.min_speed = float(self.param('min_speed', 0.02))
        self.last_steer = 0.0
        self.pub = self.create_publisher(AckermannDrive, '/ackermann_cmd_raw', 10)
        self.create_subscription(Twist, self.param('cmd_vel_topic', '/cmd_vel'), self.on_twist, 10)

    def on_twist(self, msg):
        v = msg.linear.x
        if abs(v) < self.min_speed:
            # Keep the wheels where they are; steering at standstill only scrubs.
            self.pub.publish(AckermannDrive(speed=0.0, steering_angle=self.last_steer))
            return
        self.last_steer = float(self.steering.curvature_to_steer(msg.angular.z / v))
        self.pub.publish(AckermannDrive(speed=float(v), steering_angle=self.last_steer))


def main(args=None):
    run(TwistToAckermann, args)
