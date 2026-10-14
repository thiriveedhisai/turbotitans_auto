import math
from types import SimpleNamespace as NS

from tt_obstacle_course.debug_dash import OdoCheck


class FakeHub:
    def __init__(self):
        self.state = {}
        self.notes = []

    def set(self, k, v):
        self.state[k] = v

    def note(self, msg, lvl='INFO'):
        self.notes.append(msg)


def odom(x, y, yaw):
    q = NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return NS(pose=NS(pose=NS(position=NS(x=x, y=y, z=0.0), orientation=q)))


def imu(t, wz):
    return NS(header=NS(stamp=NS(sec=int(t), nanosec=int((t % 1) * 1e9))), angular_velocity=NS(z=wz))


def test_odometry_check_reports_scale_and_turn_errors(tmp_path):
    hub = FakeHub()
    oc = OdoCheck(hub, {'odom_guard': {'odom_scale': 1.0}}, str(tmp_path), 0.3)
    oc.start({'leg1_m': 5.0, 'leg2_m': 5.0, 'turn_deg': 90})
    assert hub.state['odocheck']['active'] and hub.state['odocheck']['max_speed'] == 0.3
    # ZED reads 4% short and turns 88 deg for a true 90; the gyro turns 91 deg.
    oc.on_odom(odom(0, 0, 0))
    oc.on_imu(imu(0.0, 0.0))
    oc.mark()
    oc.on_odom(odom(4.8, 0, 0))
    oc.mark()
    for i in range(1, 101):
        oc.on_imu(imu(i * 0.01, math.radians(91)))
    oc.on_odom(odom(4.8, 0, math.radians(88)))
    oc.mark()
    oc.on_odom(odom(4.8 + 4.8 * math.cos(math.radians(88)), 4.8 * math.sin(math.radians(88)), math.radians(88)))
    st = oc.mark()
    r = st['result']
    assert not st['active']
    assert abs(r['scale'] - 5.0 / 4.8) < 1e-3
    assert abs(r['suggested_odom_scale'] - 5.0 / 4.8) < 1e-3
    assert abs(r['zed_turn_err_deg'] + 2.0) < 0.05
    assert abs(r['gyro_turn_err_deg'] - 1.0) < 0.1
    assert abs(r['path_turn_err_deg'] + 2.0) < 0.05
    assert (tmp_path / r['saved'].split('/')[-1]).exists()
