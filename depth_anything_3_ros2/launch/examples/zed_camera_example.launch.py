"""
Example launch file for ZED camera integration.

This example shows how to connect Depth Anything 3 to ZED camera topics.
Users must have zed-ros2-wrapper installed separately:
    https://github.com/stereolabs/zed-ros2-wrapper

This file provides ONLY topic remapping - no camera-specific code.
The ZED camera driver must be launched separately or included here.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """Generate launch description for ZED camera example."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_model',
            default_value='zedxm',
            description='ZED camera model (zed, zed2, zed2i, zedm, zedx, zedxm)'
        ),
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        ),
        DeclareLaunchArgument(
            'launch_zed',
            default_value='false',
            description='Whether to launch ZED camera driver (requires zed-ros2-wrapper)'
        ),

        # Optionally launch ZED camera (uncomment if zed-ros2-wrapper is installed)
        # Note: This is commented by default as it requires separate installation
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         PathJoinSubstitution([
        #             FindPackageShare('zed_wrapper'),
        #             'launch',
        #             'zed_camera.launch.py'
        #         ])
        #     ]),
        #     launch_arguments={
        #         'camera_model': LaunchConfiguration('camera_model'),
        #     }.items(),
        #     condition=IfCondition(LaunchConfiguration('launch_zed'))
        # ),

        # Launch Depth Anything 3 with ZED topic remapping
        # Default ZED topics: /zed/zed_node/rgb/image_rect_color
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            namespace='zed',
            output='screen',
            remappings=[
                ('~/image_raw', '/zed/zed_node/rgb/image_rect_color'),
                ('~/camera_info', '/zed/zed_node/rgb/camera_info'),
            ],
            parameters=[{
                'model_name': LaunchConfiguration('model_name'),
                'device': 'cuda',
                'normalize_depth': True,
                'publish_colored': True,
                'publish_confidence': True,
                'colormap': 'turbo',
                'log_inference_time': True,
            }]
        ),
    ])
