"""Record a run to a bag so tt_dash can replay it later exactly as it looked live.

  ros2 run tt_obstacle_course tt_record                 decisions, lidar, odometry, commands
  ros2 run tt_obstacle_course tt_record --camera        + camera (~10-15 MB/s raw, short runs)
  ros2 run tt_obstacle_course tt_record -o buckets_try3

Replay (truck or laptop, nothing else running):
  ros2 bag play ~/runs/bag_<name>      and in another terminal:   ros2 run tt_obstacle_course tt_dash
"""

import argparse
import os
import shutil
import sys
import time

TOPICS = [
    '/rosout', '/scan', '/odom/fused', '/odom/local', '/zed/zed_node/odom', '/zed/zed_node/imu/data',
    '/mission/state', '/mission/speed_cap', '/behavior/cmd', '/behavior/status',
    '/ackermann_cmd', '/ackermann_cmd_raw', '/race_start', '/estop', '/estop/hw_status', '/zone_event',
    '/marker/detections', '/perception/bright_exit', '/perception/left_gap',
    '/perception/right_gap', '/perception/front_gap', '/perception/gate',
    '/terrain/state', '/terrain/attitude', '/terrain/lidar_ok', '/terrain/speed_cap',
    '/odom/degraded', '/tf', '/tf_static', '/localization/status', '/localization/pose',
    '/sim/odom_truth',
]
CAMERA = ['/zed/zed_node/rgb/color/rect/image', '/zed/zed_node/rgb/color/rect/camera_info']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    ap.add_argument('-o', '--name', default=time.strftime('%m%d_%H%M%S'))
    ap.add_argument('--dir', default=os.path.expanduser('~/runs'))
    ap.add_argument('--camera', action='store_true', help='also record the camera image')
    ap.add_argument('--extra', nargs='*', default=[], help='more topics')
    a = ap.parse_args(argv)
    if shutil.which('ros2') is None:
        sys.exit('ros2 not on PATH: source the workspace first')
    os.makedirs(a.dir, exist_ok=True)
    out = os.path.join(a.dir, f'bag_{a.name}')
    topics = TOPICS + (CAMERA if a.camera else []) + a.extra
    print(f'recording {len(topics)} topics to {out}  (Ctrl-C to stop)\n'
          f'replay: ros2 bag play {out}   +   ros2 run tt_obstacle_course tt_dash', flush=True)
    os.execvp('ros2', ['ros2', 'bag', 'record', '-o', out] + topics)


if __name__ == '__main__':
    main()
