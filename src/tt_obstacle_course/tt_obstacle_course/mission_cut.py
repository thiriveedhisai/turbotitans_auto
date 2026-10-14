"""Cut-down mission for practising some zones, plus the launch command to run it.

  ros2 run tt_obstacle_course tt_mission wide buckets          -> /tmp/mission_wide_buckets.yaml
  ros2 run tt_obstacle_course tt_mission --list                zone names and their steps
"""

import argparse
import os
import sys

import yaml


def default_mission():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('tt_obstacle_course'), 'config', 'mission.yaml')
    except Exception:
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'mission.yaml')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    ap.add_argument('zones', nargs='*', help='zones to keep, in course order')
    ap.add_argument('--mission', default=default_mission())
    ap.add_argument('--list', action='store_true')
    ap.add_argument('-o', '--out')
    a = ap.parse_args(argv)
    with open(a.mission) as f:
        m = yaml.safe_load(f)
    names = [z['name'] for z in m['zones']]
    if a.list or not a.zones:
        for z in m['zones']:
            steps = ', '.join(s.get('behavior') or next(iter(k for k in s if k in ('gate', 'nav2')), '?')
                              + (' (if failed)' if s.get('only_if_failed') else '')
                              for s in z.get('steps', []))
            print(f'{z["name"]:14} timeout {z.get("timeout_s", "-"):>4}s  {steps}')
        return
    bad = [z for z in a.zones if z not in names]
    if bad:
        sys.exit(f'unknown zone(s) {bad}; zones: {" ".join(names)}')
    m['zones'] = [z for z in m['zones'] if z['name'] in a.zones]
    out = a.out or f'/tmp/mission_{"_".join(a.zones)}.yaml'
    with open(out, 'w') as f:
        yaml.safe_dump(m, f, sort_keys=False)
    first = m['zones'][0]['name']
    print(f'wrote {out}\nrun:\n  ros2 launch tt_obstacle_course obstacle_course.launch.py '
          f'mission:={out} start_zone:={first} 2>&1 | tee ~/runs/$(date +%m%d_%H%M)_{first}.log\n'
          f'start: ros2 topic pub --once /race_start std_msgs/msg/Bool "{{data: true}}"')


if __name__ == '__main__':
    main()
