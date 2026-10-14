"""Full obstacle-course stack: sensors, perception, behaviors, Nav2, governor, motor.

  ros2 launch tt_obstacle_course obstacle_course.launch.py
  ros2 launch tt_obstacle_course obstacle_course.launch.py use_nav2:=false
  ros2 launch tt_obstacle_course obstacle_course.launch.py tuning:=/path/my_tuning.yaml

Start manually (skips the green detector):
  ros2 topic pub --once /race_start std_msgs/msg/Bool "{data: true}"
E-stop:
  ros2 topic pub --once /estop std_msgs/msg/Bool "{data: true}"
Hardware e-stop (ESP8266 -> mosquitto esp/trigger) is required by default; without it
the truck will not move. Bench work without the ESP: hw_estop:=false
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('tt_obstacle_course')
    tuning = LaunchConfiguration('tuning')
    sensors = LaunchConfiguration('sensors')

    args = [
        DeclareLaunchArgument('tuning', default_value=PathJoinSubstitution([pkg, 'config', 'tuning.yaml'])),
        DeclareLaunchArgument('mission', default_value=PathJoinSubstitution([pkg, 'config', 'mission.yaml'])),
        DeclareLaunchArgument('nav2_params',
                              default_value=PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml'])),
        # No mission step sends Nav2 goals (own Hybrid-A* planner); kept as an option.
        DeclareLaunchArgument('use_nav2', default_value='false'),
        DeclareLaunchArgument('start_zone', default_value='',
                              description='Begin the mission at this zone (practice)'),
        DeclareLaunchArgument('map', default_value='',
                              description='Course map dir from tt_map (empty = no map localization)'),
        DeclareLaunchArgument('map_correct', default_value='true',
                              description='With map:=, correct /odom/fused onto the map '
                                          '(false = only localize and report)'),
        DeclareLaunchArgument('dash', default_value='false',
                              description='Start tt_dash with Drive panel, E-STOP and map making '
                                          '(http://<truck-ip>:8080)'),
        DeclareLaunchArgument('maps_dir', default_value='',
                              description='Where the dashboard saves new maps (default <workspace>/maps)'),
        DeclareLaunchArgument('hw_estop', default_value='true',
                              description='Start estop_bridge (MQTT esp/trigger) and refuse to drive '
                                          'without its RUN heartbeat'),
        DeclareLaunchArgument('sensors', default_value='true', description='Start ZED and lidar drivers'),
        DeclareLaunchArgument('camera_nodes', default_value='true',
                              description='Start start_detector and marker_detector'),
        DeclareLaunchArgument('motor', default_value='true', description='Start rc_car_controller'),
        DeclareLaunchArgument('turn_radius', default_value='',
                              description='Override the minimum turning radius (m) everywhere; '
                                          'empty = use tuning.yaml / nav2_params.yaml'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # Lidar pose relative to the ZED camera link; keep laser_offset_* in tuning.yaml equal.
        DeclareLaunchArgument('laser_x', default_value='-0.15'),
        DeclareLaunchArgument('laser_y', default_value='0.0'),
        DeclareLaunchArgument('laser_z', default_value='0.05'),
        DeclareLaunchArgument('laser_yaw', default_value='0.0'),
    ]

    drivers = GroupAction(condition=IfCondition(sensors), actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('zed_wrapper'), 'launch', 'zed_camera.launch.py'])),
            # odom_guard publishes odom -> zed_camera_link from its fused pose.
            launch_arguments={'camera_model': 'zed2i', 'publish_map_tf': 'false',
                              'publish_tf': 'false'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('sllidar_ros2'), 'launch', 'sllidar_a1_launch.py'])),
            launch_arguments={'frame_id': 'laser'}.items()),
    ])

    motor = Node(
        package='rc_car_controller', executable='rc_car_controller_node', name='rc_car_controller',
        output='screen', condition=IfCondition(LaunchConfiguration('motor')),
        parameters=[PathJoinSubstitution([FindPackageShare('rc_car_controller'), 'config', 'params.yaml']),
                    tuning])

    return LaunchDescription(args + [drivers, motor, OpaqueFunction(function=_stack_and_nav2)])


def _stack_and_nav2(context):
    pkg = FindPackageShare('tt_obstacle_course')
    tuning = LaunchConfiguration('tuning')
    nav2_params = LaunchConfiguration('nav2_params')
    use_nav2 = LaunchConfiguration('use_nav2')
    radius = LaunchConfiguration('turn_radius').perform(context).strip()
    sim_time = [{'use_sim_time': LaunchConfiguration('use_sim_time').perform(context) == 'true'}]
    ours, planner, controller = [*sim_time], [*sim_time], [*sim_time]
    if radius:
        r = float(radius)
        ours.append({'vehicle.planning_min_turn_radius': r})
        planner.append({'GridBased.minimum_turning_radius': r})
        controller.append({'FollowPath.AckermannConstraints.min_turning_r': r})

    def tt(exe, extra=(), condition=None):
        return Node(package='tt_obstacle_course', executable=exe, name=exe, output='screen',
                    parameters=[tuning, *ours, *extra], condition=condition)

    cameras = IfCondition(LaunchConfiguration('camera_nodes'))
    map_dir = LaunchConfiguration('map').perform(context).strip()
    use_map = bool(map_dir)
    correct = use_map and LaunchConfiguration('map_correct').perform(context) == 'true'
    hw_estop = LaunchConfiguration('hw_estop').perform(context) == 'true'
    stack = [
        *([tt('estop_bridge')] if hw_estop else []),
        tt('odom_guard', [{'map_correction': correct}]),
        *([tt('map_localizer', [{'map_file': map_dir}])] if use_map else []),
        tt('terrain_monitor'),
        tt('start_detector', condition=cameras),
        tt('marker_detector', condition=cameras),
        tt('course_perception'),
        tt('behavior_executor'),
        tt('twist_to_ackermann'),
        tt('speed_governor', [{'hw_estop_required': hw_estop}]),
        tt('mission_manager', [{'mission_file': LaunchConfiguration('mission'), 'use_nav2': use_nav2,
                                'start_zone': LaunchConfiguration('start_zone')}]),
    ]

    bt_xml = PathJoinSubstitution([pkg, 'config', 'navigate_ackermann.xml'])
    nav2 = GroupAction(condition=IfCondition(use_nav2), actions=[
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             output='screen', parameters=[nav2_params, *controller]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[nav2_params, *planner]),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
             output='screen', parameters=[nav2_params, *sim_time]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
             output='screen', parameters=[nav2_params, {'default_nav_to_pose_bt_xml': bt_xml},
                                          *sim_time]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[nav2_params, *sim_time]),
    ])
    dash = []
    if LaunchConfiguration('dash').perform(context) == 'true':
        maps_dir = LaunchConfiguration('maps_dir').perform(context).strip()
        dash.append(Node(package='tt_obstacle_course', executable='tt_dash', name='tt_dash',
                         output='screen', parameters=sim_time,
                         arguments=['--control'] + (['--maps-dir', maps_dir] if maps_dir else [])))
    laser_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='laser_tf',
        parameters=sim_time,
        arguments=['--x', LaunchConfiguration('laser_x'), '--y', LaunchConfiguration('laser_y'),
                   '--z', LaunchConfiguration('laser_z'), '--yaw', LaunchConfiguration('laser_yaw'),
                   '--frame-id', 'zed_camera_link', '--child-frame-id', 'laser'])
    return [laser_tf, *stack, nav2, *dash]
