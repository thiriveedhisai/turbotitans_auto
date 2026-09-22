"""Minimal PCA9685 16-channel PWM driver using smbus2.

Only the features needed by the RC car controller are implemented: setting the
output frequency and writing per-channel pulse widths (in microseconds). The
PCA9685 output-enable (OE) pin is handled separately by the ROS node via GPIO,
not by this class.
"""

import math
import time

try:
    from smbus2 import SMBus
except ImportError:  # pragma: no cover - hardware dependency
    SMBus = None


# Register map
_MODE1 = 0x00
_MODE2 = 0x01
_PRESCALE = 0xFE
_LED0_ON_L = 0x06
_ALL_LED_ON_L = 0xFA
_ALL_LED_ON_H = 0xFB
_ALL_LED_OFF_L = 0xFC
_ALL_LED_OFF_H = 0xFD

# MODE1 bits
_RESTART = 0x80
_SLEEP = 0x10
_AI = 0x20      # register auto-increment
_ALLCALL = 0x01

# MODE2 bits
_OUTDRV = 0x04  # totem-pole output
_INVRT = 0x10

_MAX_TICKS = 4096


class PCA9685:
    """Small register-level driver for the PCA9685."""

    def __init__(self, bus_num, address=0x40, osc_clock=25_000_000):
        if SMBus is None:
            raise RuntimeError(
                'smbus2 is not installed; cannot talk to the PCA9685')
        self._address = address
        self._osc_clock = osc_clock
        self._freq_hz = 50.0
        self._bus = SMBus(bus_num)
        self.reset()

    def reset(self):
        """Put the device in a known, awake state with auto-increment on."""
        # Turn all channels fully off before enabling outputs.
        self._bus.write_byte_data(self._address, _ALL_LED_ON_L, 0x00)
        self._bus.write_byte_data(self._address, _ALL_LED_ON_H, 0x00)
        self._bus.write_byte_data(self._address, _ALL_LED_OFF_L, 0x00)
        self._bus.write_byte_data(self._address, _ALL_LED_OFF_H, 0x00)
        self._bus.write_byte_data(self._address, _MODE2, _OUTDRV)
        self._bus.write_byte_data(self._address, _MODE1, _ALLCALL)
        time.sleep(0.005)
        mode1 = self._bus.read_byte_data(self._address, _MODE1)
        mode1 &= ~_SLEEP
        self._bus.write_byte_data(self._address, _MODE1, mode1 | _AI)
        time.sleep(0.005)

    def set_pwm_freq(self, freq_hz):
        """Set the PWM output frequency (Hz) for all channels."""
        self._freq_hz = float(freq_hz)
        prescale_val = self._osc_clock / (_MAX_TICKS * self._freq_hz)
        prescale = int(math.floor(prescale_val - 0.5 + 1.0))  # round
        prescale = max(3, min(255, prescale))

        old_mode = self._bus.read_byte_data(self._address, _MODE1)
        sleep_mode = (old_mode & ~_RESTART) | _SLEEP
        self._bus.write_byte_data(self._address, _MODE1, sleep_mode)
        self._bus.write_byte_data(self._address, _PRESCALE, prescale)
        self._bus.write_byte_data(self._address, _MODE1, old_mode)
        time.sleep(0.005)
        self._bus.write_byte_data(
            self._address, _MODE1, old_mode | _RESTART | _AI)

    def set_pulse_us(self, channel, pulse_us):
        """Drive ``channel`` with a pulse width of ``pulse_us`` microseconds."""
        period_us = 1_000_000.0 / self._freq_hz
        ticks = int(round(pulse_us / period_us * _MAX_TICKS))
        ticks = max(0, min(_MAX_TICKS - 1, ticks))
        self._set_pwm(channel, 0, ticks)

    def set_off(self, channel):
        """Set a channel to the full-off state (no pulse)."""
        self._set_pwm(channel, 0, 0)

    def _set_pwm(self, channel, on, off):
        base = _LED0_ON_L + 4 * channel
        data = [on & 0xFF, (on >> 8) & 0x0F, off & 0xFF, (off >> 8) & 0x0F]
        self._bus.write_i2c_block_data(self._address, base, data)

    def close(self):
        if self._bus is not None:
            self._bus.close()
            self._bus = None
