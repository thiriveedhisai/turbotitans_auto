"""Full obstacle-course stack: sensors, perception, behaviors, Nav2, governor, motor.

  ros2 launch tt_obstacle_course obstacle_course.launch.py
  ros2 launch tt_obstacle_course obstacle_course.launch.py use_nav2:=false
  ros2 launch tt_obstacle_course obstacle_course.launch.py tuning:=/path/my_tuning.yaml

Start manually (skips the green detector):
  ros2 topic pub --once /race_start std_msgs/msg/Bool "{data: true}"
E-stop:
  ros2 topic pub --once /estop std_msgs/msg/Bool "{data: true}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('tt_obstacle_course')
    tuning = LaunchConfiguration('tuning')
    nav2_params = LaunchConfiguration('nav2_params')
    use_nav2 = LaunchConfiguration('use_nav2')
    sensors = LaunchConfiguration('sensors')

    args = [
        DeclareLaunchArgument('tuning', default_value=PathJoinSubstitution([pkg, 'config', 'tuning.yaml'])),
        DeclareLaunchArgument('mission', default_value=PathJoinSubstitution([pkg, 'config', 'mission.yaml'])),
        DeclareLaunchArgument('nav2_params',
                              default_value=PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml'])),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('sensors', default_value='true', description='Start ZED and lidar drivers'),
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
            launch_arguments={'camera_model': 'zed2i', 'publish_map_tf': 'false'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('sllidar_ros2'), 'launch', 'sllidar_a1_launch.py'])),
            launch_arguments={'frame_id': 'laser'}.items()),
    ])

    laser_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='laser_tf',
        arguments=['--x', LaunchConfiguration('laser_x'), '--y', LaunchConfiguration('laser_y'),
                   '--z', LaunchConfiguration('laser_z'), '--yaw', LaunchConfiguration('laser_yaw'),
                   '--frame-id', 'zed_camera_link', '--child-frame-id', 'laser'])

    motor = Node(
        package='rc_car_controller', executable='rc_car_controller_node', name='rc_car_controller',
        output='screen',
        parameters=[PathJoinSubstitution([FindPackageShare('rc_car_controller'), 'config', 'params.yaml']),
                    tuning])

    def tt(exe, extra=()):
        return Node(package='tt_obstacle_course', executable=exe, name=exe, output='screen',
                    parameters=[tuning, *extra])

    stack = [
        tt('terrain_monitor'),
        tt('start_detector'),
        tt('marker_detector'),
        tt('course_perception'),
        tt('behavior_executor'),
        tt('twist_to_ackermann'),
        tt('speed_governor'),
        tt('mission_manager', [{'mission_file': LaunchConfiguration('mission'), 'use_nav2': use_nav2}]),
    ]

    bt_xml = PathJoinSubstitution([pkg, 'config', 'navigate_ackermann.xml'])
    nav2 = GroupAction(condition=IfCondition(use_nav2), actions=[
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             output='screen', parameters=[nav2_params]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[nav2_params]),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
             output='screen', parameters=[nav2_params]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
             output='screen', parameters=[nav2_params, {'default_nav_to_pose_bt_xml': bt_xml}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen', parameters=[nav2_params]),
    ])

    return LaunchDescription(args + [drivers, laser_tf, motor, *stack, nav2])
