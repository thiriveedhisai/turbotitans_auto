import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/turbotitans/ros2_ws/src/rc_car_controller/install/rc_car_controller'
