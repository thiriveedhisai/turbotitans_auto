from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from tt_obstacle_course.speed_governor import SpeedGovernor  # noqa: E402


def Bool(data):
    return SimpleNamespace(data=data)


class _Log:
    def warn(self, *_):
        pass


def _gov(required=True):
    g = SimpleNamespace(estop=False, hw_required=required, hw_latch=True, hw_stale=1.0,
                        hw_stop=True, hw_t=-1e9, hw_seen_run=False, hw_latched=False, t=0.0)
    g.now_s = lambda: g.t
    g.get_logger = _Log
    return g


def hw(g, stop):
    SpeedGovernor.on_hw_estop(g, Bool(data=stop))


def stopped(g):
    return SpeedGovernor.stopped(g, g.t)


def test_required_without_bridge_stops_optional_does_not():
    assert stopped(_gov(True))
    assert not stopped(_gov(False))


def test_boot_stop_not_latched_then_run_then_latch_until_release():
    g = _gov()
    hw(g, True)
    assert stopped(g) and not g.hw_latched
    hw(g, False)
    assert not stopped(g)
    hw(g, True)
    hw(g, False)
    assert g.hw_latched and stopped(g)
    SpeedGovernor.on_estop(g, Bool(data=False))
    assert not stopped(g)


def test_release_ignored_while_button_at_stop():
    g = _gov()
    hw(g, False)
    hw(g, True)
    SpeedGovernor.on_estop(g, Bool(data=False))
    assert g.hw_latched and stopped(g)


def test_heartbeat_silence_stops_when_required():
    g = _gov()
    hw(g, False)
    g.t = 2.0
    assert stopped(g)
