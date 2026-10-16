"""Hardware E-stop: ESP8266 over MQTT -> /estop/hw (true = stop).

The ESP publishes {"pin_state": int, ...} on esp/trigger to the local mosquitto broker.
Only pin_state counts: STOP when it equals stop_pin_state (0 = button pressed on this ESP),
RUN otherwise; other fields (heartbeat) are ignored. STOP until the first message arrives;
after that the last pin_state holds. speed_governor latches a STOP.
"""

import json
import threading

from std_msgs.msg import Bool, String

from tt_obstacle_course.ros_common import TTNode, run


class EstopBridge(TTNode):

    def __init__(self):
        super().__init__('estop_bridge')
        self.host = str(self.param('mqtt_host', 'localhost'))
        self.port = int(self.param('mqtt_port', 1883))
        self.topic = str(self.param('mqtt_topic', 'esp/trigger'))
        self.stop_pin = int(self.param('stop_pin_state', 0))

        self.lock = threading.Lock()
        self.pin_state = None
        self.connected = False
        self.stop = None

        self.pub = self.create_publisher(Bool, '/estop/hw', 10)
        self.status_pub = self.create_publisher(String, '/estop/hw_status', 10)
        self.client = self.connect()
        # 20 Hz: a button press reaches the governor (also 20 Hz) within ~0.1 s.
        self.create_timer(0.05, self.tick)

    def connect(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.get_logger().error('E-STOP: paho-mqtt missing (sudo apt install python3-paho-mqtt); '
                                    'hardware e-stop reads as STOP')
            return None
        # paho 2.x needs the callback API version; 1.x has no such argument.
        api = getattr(mqtt, 'CallbackAPIVersion', None)
        client = mqtt.Client(api.VERSION1) if api else mqtt.Client()
        client.on_connect = self.on_connect
        client.on_disconnect = lambda *a: setattr(self, 'connected', False)
        client.on_message = self.on_message
        client.reconnect_delay_set(1, 5)
        client.connect_async(self.host, self.port, 30)
        client.loop_start()
        self.get_logger().info(f'E-STOP: connecting to mqtt {self.host}:{self.port} topic {self.topic}')
        return client

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            client.subscribe(self.topic)
            self.get_logger().info(f'E-STOP: mqtt connected, listening on {self.topic}')
        else:
            self.get_logger().error(f'E-STOP: mqtt connect failed rc={rc}')

    def on_message(self, client, userdata, msg):
        try:
            pin = int(json.loads(msg.payload.decode('utf-8'))['pin_state'])
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            self.get_logger().error(f'E-STOP: bad payload {msg.payload[:80]!r}: {e}')
            return
        with self.lock:
            changed = pin != self.pin_state
            self.pin_state = pin
        if changed:
            self.get_logger().info(f'E-STOP: ESP pin_state = {pin} '
                                   f'({"STOP" if pin == self.stop_pin else "RUN"})')

    def tick(self):
        with self.lock:
            pin = self.pin_state
        if pin is None:
            stop, reason = True, 'no mqtt broker' if not self.connected else 'waiting for e-stop'
        else:
            stop = pin == self.stop_pin
            reason = 'button' if stop else 'ok'
        if stop != self.stop:
            if stop:
                self.get_logger().warn(f'E-STOP: hardware STOP ({reason})')
            else:
                self.get_logger().info(f'E-STOP: hardware RUN ({reason})')
            self.stop = stop
        self.pub.publish(Bool(data=stop))
        self.status_pub.publish(String(data=json.dumps(
            {'stop': stop, 'reason': reason, 'pin_state': pin})))

    def destroy_node(self):
        if self.client is not None:
            self.client.loop_stop()
        super().destroy_node()


def main(args=None):
    run(EstopBridge, args)
