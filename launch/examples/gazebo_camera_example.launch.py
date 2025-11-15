"""
Example launch file for Gazebo simulated camera integration.

This example shows how to connect Depth Anything 3 to Gazebo camera topics.
Useful for testing in simulation before deploying to real hardware.

This file provides ONLY topic remapping - no camera-specific code.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Gazebo camera example."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Gazebo camera image topic'
        ),
        DeclareLaunchArgument(
            'camera_namespace',
            default_value='camera',
            description='Camera namespace in Gazebo'
        ),

        # Launch Depth Anything 3 with Gazebo topic remapping
        # Default Gazebo camera topics: /camera/image_raw
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            namespace=LaunchConfiguration('camera_namespace'),
            output='screen',
            remappings=[
                ('~/image_raw', LaunchConfiguration('camera_topic')),
                ('~/camera_info', '/camera/camera_info'),
            ],
            parameters=[{
                'model_name': LaunchConfiguration('model_name'),
                'device': 'cuda',  # Use CPU for lighter simulation loads
                'normalize_depth': True,
                'publish_colored': True,
                'publish_confidence': True,
                'colormap': 'turbo',
                'log_inference_time': True,
            }]
        ),
    ])
