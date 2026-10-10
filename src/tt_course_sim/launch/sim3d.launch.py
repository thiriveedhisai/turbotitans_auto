"""Full stack in the 3-D Gazebo Harmonic world (headless, GPU rendering).

  ros2 launch tt_course_sim sim3d.launch.py
  ros2 launch tt_course_sim sim3d.launch.py seed:=3 rails:=false turn_radius:=1.2
  ros2 launch tt_course_sim sim3d.launch.py start_zone:=wide

Unlike the 2-D sim, the camera nodes run for real here: start_detector must see
the green light, marker_detector reads the AprilTags. Ends when the referee
finishes the run; results in output_dir.
"""

import os

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from tt_course_sim.gz_world import generate


def _tag_map(tuning_path):
    with open(tuning_path, encoding='utf-8') as f:
        md = yaml.safe_load(f)['marker_detector']['ros__parameters']
    return dict(zip(md['tag_zones'], md['tag_ids']))


def _setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    seed, rails, radius = int(lc('seed')), lc('rails') == 'true', float(lc('turn_radius'))
    start_zone, out = lc('start_zone'), lc('output_dir')
    share = FindPackageShare('tt_obstacle_course').perform(context)
    tuning = lc('tuning') or os.path.join(share, 'config', 'tuning.yaml')
    world, course, spawn = generate(os.path.join(out, 'world'), seed=seed, rails=rails,
                                    radius=radius, start_zone=start_zone, tag_map=_tag_map(tuning),
                                    friction=float(lc('friction')))
    sim_time = {'use_sim_time': True}

    gz = ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', '--headless-rendering', world],
                        output='screen')
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge', output='screen',
        parameters=[sim_time],
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                   '/gz/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                   '/gz/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
                   '/gz/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                   '/gz/zed/image@sensor_msgs/msg/Image[gz.msgs.Image',
                   '/gz/zed/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                   '/gz/zed/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                   '/gz/chase@sensor_msgs/msg/Image[gz.msgs.Image',
                   '/gz_cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'],
        remappings=[('/gz/scan', '/scan'), ('/gz/imu', '/zed/zed_node/imu/data'),
                    ('/gz/odom', '/gz/odom_world'),
                    ('/gz/zed/image', '/zed/zed_node/rgb/color/rect/image'),
                    ('/gz/zed/camera_info', '/zed/zed_node/rgb/color/rect/camera_info'),
                    ('/gz/zed/depth_image', '/zed/zed_node/depth/depth_registered')])
    vo = lc('vo') == 'true'
    drive = Node(package='tt_course_sim', executable='ackermann_to_gz', name='ackermann_to_gz',
                 output='screen', parameters=[sim_time, {'turn_radius': radius, 'publish_odom': not vo}])
    # ZED odometry from the simulated camera + IMU (drifts, loses tracking like the real one).
    zed_vo = Node(package='tt_course_sim', executable='zed_vo_sim', name='zed_vo_sim', output='screen',
                  parameters=[sim_time, {'seed': seed, 'rails': rails, 'spawn': list(spawn),
                                         'blackout_zones': lc('vo_blackout')}])
    referee = Node(package='tt_course_sim', executable='gz_referee', name='gz_referee',
                   output='screen',
                   parameters=[sim_time, {'seed': seed, 'rails': rails, 'start_zone': start_zone,
                                          'spawn': list(spawn), 'output_dir': out,
                                          'timeout': float(lc('timeout'))}])
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'obstacle_course.launch.py')),
        launch_arguments={'sensors': 'false', 'camera_nodes': 'true', 'motor': 'false',
                          'use_sim_time': 'true', 'use_nav2': lc('use_nav2'),
                          'turn_radius': str(radius), 'start_zone': start_zone,
                          'tuning': tuning}.items())
    stop = [RegisterEventHandler(OnProcessExit(target_action=a, on_exit=[EmitEvent(event=Shutdown())]))
            for a in (referee, gz)]
    return [gz, bridge, drive, *([zed_vo] if vo else []), referee, stack, *stop]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('rails', default_value='true'),
        DeclareLaunchArgument('turn_radius', default_value='1.52'),
        DeclareLaunchArgument('friction', default_value='0.8',
                              description='Course surface friction (painted plywood ~0.8, dusty ~0.6)'),
        DeclareLaunchArgument('start_zone', default_value=''),
        DeclareLaunchArgument('vo', default_value='true',
                              description='ZED odometry from simulated camera VO (false = ground truth)'),
        DeclareLaunchArgument('vo_blackout', default_value='tunnel,car_wash',
                              description='Zones where visual tracking is forced lost (comma list)'),
        DeclareLaunchArgument('use_nav2', default_value='false'),
        DeclareLaunchArgument('timeout', default_value='600.0'),
        DeclareLaunchArgument('output_dir', default_value='/tmp/tt_gz_run'),
        DeclareLaunchArgument('tuning', default_value=''),
        OpaqueFunction(function=_setup),
    ])
