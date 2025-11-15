"""
Example launch file for OAK-D camera integration.

This example shows how to connect Depth Anything 3 to OAK-D camera topics.
Users must have depthai-ros installed separately:
    https://github.com/luxonis/depthai-ros

This file provides ONLY topic remapping - no camera-specific code.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for OAK-D camera example."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        ),
        DeclareLaunchArgument(
            'camera_name',
            default_value='oak',
            description='OAK-D camera name'
        ),

        # Launch Depth Anything 3 with OAK-D topic remapping
        # Default OAK-D RGB topic: /oak/rgb/image_raw
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            namespace='oak',
            output='screen',
            remappings=[
                ('~/image_raw', '/oak/rgb/image_raw'),
                ('~/camera_info', '/oak/rgb/camera_info'),
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
