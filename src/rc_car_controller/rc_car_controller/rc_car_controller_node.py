"""ROS 2 node: AckermannDrive -> PCA9685 servo/ESC PWM for an RC car.

Hardware assumptions (all overridable via parameters):
  * PCA9685 on I2C bus 7 at address 0x40, 50 Hz output.
  * Channel 0 -> steering servo, channel 3 -> motor ESC.
  * PCA9685 OE pin -> Jetson Nano 40-pin header GPIO09 (BOARD pin 7).

Startup sequence (as required):
  1. OE is driven HIGH immediately -> PCA9685 outputs disabled.
  2. PCA9685 is initialised: 50 Hz, servo + motor set to 1500 us (neutral).
  3. OE is driven LOW -> outputs enabled with the safe neutral values already
     latched, so the servo/ESC never see a glitch on power-up.
"""

import math

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor

from rc_car_controller.pca9685 import PCA9685

try:
    import Jetson.GPIO as GPIO
except Exception:  # pragma: no cover - only available on the Jetson
    GPIO = None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _interp(x, xs, ys):
    """Piecewise-linear lookup; xs ascending, clamps at the ends."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


class RcCarController(Node):
    def __init__(self):
        super().__init__('rc_car_controller')

        # --- I2C / PCA9685 ---
        self.declare_parameter('i2c_bus', 7)
        self.declare_parameter('i2c_address', 0x40)
        self.declare_parameter('pwm_freq_hz', 50.0)
        self.declare_parameter('servo_channel', 0)
        self.declare_parameter('motor_channel', 3)

        # --- OE pin (Jetson Nano header GPIO09 = BOARD pin 7) ---
        self.declare_parameter('oe_gpio_mode', 'BOARD')
        self.declare_parameter('oe_gpio_pin', 7)

        # --- Steering (servo) mapping, microseconds ---
        self.declare_parameter('servo_center_us', 1500)
        self.declare_parameter('servo_min_us', 1000)
        self.declare_parameter('servo_max_us', 2000)
        self.declare_parameter('max_steering_angle', 0.5)  # rad -> full travel
        self.declare_parameter('steering_direction', 1)    # -1 to invert

        # --- Throttle (ESC) mapping, microseconds ---
        self.declare_parameter('motor_neutral_us', 1500)
        self.declare_parameter('motor_min_us', 1000)
        self.declare_parameter('motor_max_us', 2000)
        self.declare_parameter('max_speed', 2.0)           # m/s -> full throttle
        self.declare_parameter('motor_direction', 1)       # -1 to invert

        # --- I/O ---
        self.declare_parameter('input_topic', 'ackermann_cmd')
        self.declare_parameter('command_timeout', 0.5)     # s, failsafe

        # --- Calibration (defaults keep the original linear behaviour) ---
        self.declare_parameter('pca_osc_clock_hz', 25_000_000)
        self.declare_parameter('steer_trim_us', 0)
        self.declare_parameter('speed_map_mode', 'linear')  # linear | table
        table = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('fwd_speed_mps', [0.0, 2.0], table)
        self.declare_parameter('fwd_pulse_us', [1500.0, 2000.0], table)
        self.declare_parameter('rev_speed_mps', [0.0, -2.0], table)
        self.declare_parameter('rev_pulse_us', [1500.0, 1000.0], table)
        self.declare_parameter('esc_deadband_us', 0)
        self.declare_parameter('esc_reverse_mode', 'direct')  # direct | double_tap
        self.declare_parameter('esc_reverse_brake_ms', 120)
        self.declare_parameter('esc_reverse_neutral_ms', 120)
        self.declare_parameter('speed_scale', 1.0)

        # --- Closed-loop speed on odometry (no wheel encoders) ---
        self.declare_parameter('speed_feedback_enabled', False)
        self.declare_parameter('speed_feedback_topic', '/zed/zed_node/odom')
        self.declare_parameter('speed_kp_us_per_mps', 40.0)
        self.declare_parameter('speed_ki_us_per_m', 25.0)
        self.declare_parameter('speed_i_limit_us', 80.0)
        self.declare_parameter('speed_correction_limit_us', 150.0)

        gp = self.get_parameter
        self._bus_num = gp('i2c_bus').value
        self._address = gp('i2c_address').value
        self._freq = gp('pwm_freq_hz').value
        self._servo_ch = gp('servo_channel').value
        self._motor_ch = gp('motor_channel').value

        self._oe_mode = gp('oe_gpio_mode').value
        self._oe_pin = gp('oe_gpio_pin').value

        self._servo_center = gp('servo_center_us').value
        self._servo_min = gp('servo_min_us').value
        self._servo_max = gp('servo_max_us').value
        self._max_steer = gp('max_steering_angle').value
        self._steer_dir = gp('steering_direction').value

        self._motor_neutral = gp('motor_neutral_us').value
        self._motor_min = gp('motor_min_us').value
        self._motor_max = gp('motor_max_us').value
        self._max_speed = gp('max_speed').value
        self._motor_dir = gp('motor_direction').value

        self._timeout = gp('command_timeout').value
        self._last_cmd_time = self.get_clock().now()

        self._servo_neutral = int(self._servo_center + gp('steer_trim_us').value)
        self._speed_mode = gp('speed_map_mode').value
        self._fwd_v = [float(v) for v in gp('fwd_speed_mps').value]
        self._fwd_us = [float(v) for v in gp('fwd_pulse_us').value]
        rev_v = [abs(float(v)) for v in gp('rev_speed_mps').value]
        rev_us = [float(v) for v in gp('rev_pulse_us').value]
        order = sorted(range(len(rev_v)), key=lambda i: rev_v[i])
        self._rev_v = [rev_v[i] for i in order]
        self._rev_us = [rev_us[i] for i in order]
        self._check_table('fwd', self._fwd_v, self._fwd_us)
        self._check_table('rev', self._rev_v, self._rev_us)
        self._deadband = gp('esc_deadband_us').value
        self._double_tap = gp('esc_reverse_mode').value == 'double_tap'
        self._brake_s = gp('esc_reverse_brake_ms').value / 1000.0
        self._rev_neutral_s = gp('esc_reverse_neutral_ms').value / 1000.0
        self._speed_scale = gp('speed_scale').value

        # ESC direction state: +1 last drove forward, -1 reverse engaged.
        self._esc_dir = 0
        self._rev_seq_start = None

        self._fb_enabled = gp('speed_feedback_enabled').value
        self._kp = gp('speed_kp_us_per_mps').value
        self._ki = gp('speed_ki_us_per_m').value
        self._i_limit = gp('speed_i_limit_us').value
        self._corr_limit = gp('speed_correction_limit_us').value
        self._integral = 0.0
        self._last_pi_time = None
        self._meas_speed = 0.0
        self._meas_time = None
        self._prev_odom = None

        # 1) OE HIGH -> outputs disabled before we touch the PCA9685.
        self._gpio_ready = self._setup_oe_high()

        # 2) Initialise PCA9685 and latch neutral values.
        self._pca = PCA9685(self._bus_num, self._address,
                            osc_clock=gp('pca_osc_clock_hz').value)
        self._pca.set_pwm_freq(self._freq)
        self._pca.set_pulse_us(self._servo_ch, self._servo_neutral)
        self._pca.set_pulse_us(self._motor_ch, self._motor_neutral)
        self.get_logger().info(
            f'PCA9685 initialised on bus {self._bus_num} @ '
            f'0x{self._address:02X}, {self._freq:.0f} Hz, '
            f'servo ch{self._servo_ch}/motor ch{self._motor_ch} at neutral.')

        # 3) OE LOW -> enable outputs now that neutral is latched.
        self._enable_outputs()

        self._sub = self.create_subscription(
            AckermannDrive, gp('input_topic').value, self._on_cmd, 10)
        if self._fb_enabled:
            self._odom_sub = self.create_subscription(
                Odometry, gp('speed_feedback_topic').value, self._on_odom, 10)
        self._watchdog = self.create_timer(0.1, self._check_timeout)
        self.get_logger().info(
            f"Listening for AckermannDrive on '{gp('input_topic').value}'.")

    # ------------------------------------------------------------------ OE pin
    def _setup_oe_high(self):
        if GPIO is None:
            self.get_logger().warn(
                'Jetson.GPIO not available; OE pin will not be controlled. '
                'Ensure the PCA9685 OE is wired/pulled correctly.')
            return False
        mode = getattr(GPIO, self._oe_mode, GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setmode(mode)
        # initial=HIGH keeps outputs disabled from the very first instant.
        GPIO.setup(self._oe_pin, GPIO.OUT, initial=GPIO.HIGH)
        self.get_logger().info(
            f'OE held HIGH (outputs disabled) on {self._oe_mode} '
            f'pin {self._oe_pin}.')
        return True

    def _enable_outputs(self):
        if self._gpio_ready:
            GPIO.output(self._oe_pin, GPIO.LOW)
            self.get_logger().info('OE driven LOW -> PCA9685 outputs enabled.')

    def _disable_outputs(self):
        if self._gpio_ready:
            GPIO.output(self._oe_pin, GPIO.HIGH)

    # ------------------------------------------------------------- conversions
    def _check_table(self, name, xs, ys):
        if len(xs) < 2 or len(xs) != len(ys) or any(
                b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError(
                f'{name} speed table needs >= 2 strictly increasing |speed| '
                f'points with matching pulse entries')

    def _steering_to_us(self, angle_rad):
        angle = _clamp(angle_rad * self._steer_dir,
                       -self._max_steer, self._max_steer)
        frac = angle / self._max_steer if self._max_steer else 0.0
        if frac >= 0.0:
            us = self._servo_neutral + frac * (self._servo_max - self._servo_neutral)
        else:
            us = self._servo_neutral + frac * (self._servo_neutral - self._servo_min)
        return int(round(_clamp(us, self._servo_min, self._servo_max)))

    def _speed_to_us(self, speed, correction_us=0.0):
        v = speed * self._speed_scale
        if v == 0.0:
            return int(self._motor_neutral)
        if self._speed_mode == 'table':
            # Table holds raw measured pulses, so motor_direction is not applied.
            if v > 0.0:
                offset = _interp(v, self._fwd_v, self._fwd_us) - self._motor_neutral
            else:
                offset = _interp(-v, self._rev_v, self._rev_us) - self._motor_neutral
            direction = 1
        else:
            value = _clamp(v, -self._max_speed, self._max_speed)
            frac = value / self._max_speed if self._max_speed else 0.0
            span = (self._motor_max - self._motor_neutral if frac > 0.0
                    else self._motor_neutral - self._motor_min)
            offset = (math.copysign(self._deadband, frac)
                      + frac * (span - self._deadband))
            direction = self._motor_dir
        offset += correction_us
        # Feedback may trim throttle but never flip the drive direction.
        offset = max(offset, 0.0) if v > 0.0 else min(offset, 0.0)
        us = self._motor_neutral + offset * direction
        return int(round(_clamp(us, self._motor_min, self._motor_max)))

    def _speed_correction(self, target, now):
        fresh = self._meas_time is not None and now - self._meas_time < 0.3
        if not self._fb_enabled or not fresh or abs(target) < 0.05:
            self._integral = 0.0
            self._last_pi_time = None
            return 0.0
        err = target - self._meas_speed
        if self._last_pi_time is not None:
            self._integral += err * min(now - self._last_pi_time, 0.2)
            if self._ki > 0.0:
                lim = self._i_limit / self._ki
                self._integral = _clamp(self._integral, -lim, lim)
        self._last_pi_time = now
        return _clamp(self._kp * err + self._ki * self._integral,
                      -self._corr_limit, self._corr_limit)

    def _motor_pulse(self, target, now):
        if target < 0.0 and self._double_tap and self._esc_dir != -1:
            # Hobby ESCs brake on the first reverse pulse after forward:
            # reverse (brake) -> neutral -> reverse actually engages it.
            if self._rev_seq_start is None:
                self._rev_seq_start = now
                self._integral = 0.0
            t = now - self._rev_seq_start
            if t < self._brake_s:
                return self._speed_to_us(target)
            if t < self._brake_s + self._rev_neutral_s:
                return int(self._motor_neutral)
            self._esc_dir = -1
            self._rev_seq_start = None
        elif target > 0.0:
            if self._esc_dir != 1:
                self._integral = 0.0
            self._esc_dir = 1
            self._rev_seq_start = None
        else:
            self._rev_seq_start = None
        return self._speed_to_us(target, self._speed_correction(target, now))

    # -------------------------------------------------------------- callbacks
    def _now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _on_odom(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self._prev_odom is not None:
            dt = stamp - self._prev_odom[0]
            if dt > 1e-3:
                v = ((p.x - self._prev_odom[1]) * math.cos(yaw)
                     + (p.y - self._prev_odom[2]) * math.sin(yaw)) / dt
                self._meas_speed = 0.6 * self._meas_speed + 0.4 * v
        self._prev_odom = (stamp, p.x, p.y)
        self._meas_time = self._now_s()

    def _on_cmd(self, msg):
        self._last_cmd_time = self.get_clock().now()
        servo_us = self._steering_to_us(msg.steering_angle)
        motor_us = self._motor_pulse(msg.speed, self._now_s())
        self._pca.set_pulse_us(self._servo_ch, servo_us)
        self._pca.set_pulse_us(self._motor_ch, motor_us)

    def _check_timeout(self):
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
        if elapsed > self._timeout:
            # Failsafe: no fresh command -> return to neutral (stop/straight).
            self._rev_seq_start = None
            self._integral = 0.0
            self._pca.set_pulse_us(self._servo_ch, self._servo_neutral)
            self._pca.set_pulse_us(self._motor_ch, self._motor_neutral)

    # --------------------------------------------------------------- shutdown
    def safe_stop(self):
        try:
            self._pca.set_pulse_us(self._servo_ch, self._servo_neutral)
            self._pca.set_pulse_us(self._motor_ch, self._motor_neutral)
            self._disable_outputs()
            self._pca.close()
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'Error during safe stop: {exc}')
        if self._gpio_ready:
            GPIO.cleanup()


def main(args=None):
    rclpy.init(args=args)
    node = RcCarController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.safe_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
