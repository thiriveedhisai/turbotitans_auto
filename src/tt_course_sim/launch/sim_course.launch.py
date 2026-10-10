"""Full obstacle-course stack driving the 2-D course simulator.

  ros2 launch tt_course_sim sim_course.launch.py
  ros2 launch tt_course_sim sim_course.launch.py use_nav2:=false rails:=false seed:=3

The launch shuts down when the run ends (finish, fall, stuck or timeout);
results land in output_dir (result.json, trajectory.csv, run.png).
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('rails', default_value='true'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('timeout', default_value='600.0'),
        DeclareLaunchArgument('true_min_turn_radius', default_value='1.52'),
        DeclareLaunchArgument('output_dir', default_value='/tmp/tt_sim'),
        DeclareLaunchArgument('start_zone', default_value='',
                              description='Spawn at this zone and start the mission there'),
        DeclareLaunchArgument('tuning', default_value=PathJoinSubstitution(
            [FindPackageShare('tt_obstacle_course'), 'config', 'tuning.yaml'])),
    ]
    sim = Node(
        package='tt_course_sim', executable='course_sim', name='course_sim', output='screen',
        parameters=[{
            'rails': LaunchConfiguration('rails'),
            'seed': LaunchConfiguration('seed'),
            'timeout': LaunchConfiguration('timeout'),
            'true_min_turn_radius': LaunchConfiguration('true_min_turn_radius'),
            'output_dir': LaunchConfiguration('output_dir'),
            'start_zone': LaunchConfiguration('start_zone'),
        }])
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('tt_obstacle_course'), 'launch', 'obstacle_course.launch.py'])),
        launch_arguments={'sensors': 'false', 'camera_nodes': 'false', 'motor': 'false',
                          'use_nav2': LaunchConfiguration('use_nav2'),
                          'turn_radius': LaunchConfiguration('true_min_turn_radius'),
                          'start_zone': LaunchConfiguration('start_zone'),
                          'tuning': LaunchConfiguration('tuning')}.items())
    stop_all = RegisterEventHandler(OnProcessExit(target_action=sim,
                                                  on_exit=[EmitEvent(event=Shutdown())]))
    return LaunchDescription(args + [sim, stack, stop_all])
