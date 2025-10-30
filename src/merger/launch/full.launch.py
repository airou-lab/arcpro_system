import os
import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    twist_to_ackermann = Node(
        package='twist_to_ackermann',
        executable='twist_to_ackermann',
        name='twist_to_ackermann',
        parameters=[
            {'wheelbase': 0.4826},#manually measured wheelbase was 9.5in
            {'use_stamps': False}
        ],
        remappings=[
            ('/cmd_vel', '/cmd_vel_nav'),
            ('/ackermann_cmd', '/drive')
        ]
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
#        arguments=['-d', os.path.join(
#            FindPackageShare('waypoint_with_map').find('waypoint_with_map'),
#            'rviz',
#            'waypointer.rviz'  # your rviz config file
#        )],
        output='screen'
    )
    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[os.path.join("merger", 'config/ekf.yaml'), {'use_sim_time': LaunchConfiguration('use_sim_time')}]
        )


    return LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='use_sim_time', default_value='false'),
        # nav2\
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('nav2_bringup'),
                    'launch',
                    'bringup_launch.py'
                ])
            ),
            launch_arguments={
                'use_collision_monitor': 'False',
                'params_file': PathJoinSubstitution([
                   FindPackageShare('merger'),
                   'config',
                   'nav2.yaml'
                ]),
                'use_sim_time': 'false',
                'use_docking': 'False' ,
            }.items()
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('f1tenth_stack'),
                'launch',
                'no_lidar_bringup_launch.py'
            ])
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ),
        #ldiar fixme, currently doesnt work with the stack, need to bringup seperatly..
#ros2 launch launch ydlidar_launch.py
        # IncludeLaunchDescription(
        #     PathJoinSubstitution([
        #         FindPackageShare('ydlidar_ros2_driver'),
        #         'launch',
        #         'ydlidar_launch.py'
        #     ])
        # ),
        # vsec depracted..
        # IncludeLaunchDescription(
        #     PathJoinSubstitution([
        #         FindPackageShare('launches'),
        #         'launch',
        #         'vesc.launch.py'
        #     ])
        # ),
        rviz_node,
        twist_to_ackermann,
        robot_localization_node
    ])
