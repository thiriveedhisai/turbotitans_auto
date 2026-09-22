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

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive

from rc_car_controller.pca9685 import PCA9685

try:
    import Jetson.GPIO as GPIO
except Exception:  # pragma: no cover - only available on the Jetson
    GPIO = None


def _clamp(value, low, high):
    return max(low, min(high, value))


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

        # 1) OE HIGH -> outputs disabled before we touch the PCA9685.
        self._gpio_ready = self._setup_oe_high()

        # 2) Initialise PCA9685 and latch neutral values.
        self._pca = PCA9685(self._bus_num, self._address)
        self._pca.set_pwm_freq(self._freq)
        self._pca.set_pulse_us(self._servo_ch, self._servo_center)
        self._pca.set_pulse_us(self._motor_ch, self._motor_neutral)
        self.get_logger().info(
            f'PCA9685 initialised on bus {self._bus_num} @ '
            f'0x{self._address:02X}, {self._freq:.0f} Hz, '
            f'servo ch{self._servo_ch}/motor ch{self._motor_ch} at neutral.')

        # 3) OE LOW -> enable outputs now that neutral is latched.
        self._enable_outputs()

        self._sub = self.create_subscription(
            AckermannDrive, gp('input_topic').value, self._on_cmd, 10)
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
    def _steering_to_us(self, angle_rad):
        angle = _clamp(angle_rad * self._steer_dir,
                       -self._max_steer, self._max_steer)
        frac = angle / self._max_steer if self._max_steer else 0.0
        if frac >= 0.0:
            us = self._servo_center + frac * (self._servo_max - self._servo_center)
        else:
            us = self._servo_center + frac * (self._servo_center - self._servo_min)
        return int(round(_clamp(us, self._servo_min, self._servo_max)))

    def _speed_to_us(self, speed):
        value = _clamp(speed * self._motor_dir,
                       -self._max_speed, self._max_speed)
        frac = value / self._max_speed if self._max_speed else 0.0
        if frac >= 0.0:
            us = self._motor_neutral + frac * (self._motor_max - self._motor_neutral)
        else:
            us = self._motor_neutral + frac * (self._motor_neutral - self._motor_min)
        return int(round(_clamp(us, self._motor_min, self._motor_max)))

    # -------------------------------------------------------------- callbacks
    def _on_cmd(self, msg):
        self._last_cmd_time = self.get_clock().now()
        servo_us = self._steering_to_us(msg.steering_angle)
        motor_us = self._speed_to_us(msg.speed)
        self._pca.set_pulse_us(self._servo_ch, servo_us)
        self._pca.set_pulse_us(self._motor_ch, motor_us)

    def _check_timeout(self):
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
        if elapsed > self._timeout:
            # Failsafe: no fresh command -> return to neutral (stop/straight).
            self._pca.set_pulse_us(self._servo_ch, self._servo_center)
            self._pca.set_pulse_us(self._motor_ch, self._motor_neutral)

    # --------------------------------------------------------------- shutdown
    def safe_stop(self):
        try:
            self._pca.set_pulse_us(self._servo_ch, self._servo_center)
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
