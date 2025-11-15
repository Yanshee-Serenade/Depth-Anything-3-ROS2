"""
Example launch file for testing with static images using image_publisher.

This example shows how to test Depth Anything 3 with static images.
Users must have image_publisher installed:
    sudo apt install ros-humble-image-publisher

Usage:
    ros2 launch depth_anything_3_ros2 image_publisher_test.launch.py \
        image_path:=/path/to/your/image.jpg
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for image publisher test."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_path',
            default_value='',
            description='Path to test image file (jpg, png, etc.)'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='1.0',
            description='Image publish rate in Hz'
        ),
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        ),

        # Launch image_publisher node
        Node(
            package='image_publisher',
            executable='image_publisher_node',
            name='image_publisher',
            parameters=[{
                'filename': LaunchConfiguration('image_path'),
                'publish_rate': LaunchConfiguration('publish_rate'),
                'frame_id': 'camera_optical_frame',
            }],
            remappings=[
                ('image_raw', '/test_image/image_raw'),
            ]
        ),

        # Launch Depth Anything 3 node
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            namespace='test',
            output='screen',
            remappings=[
                ('~/image_raw', '/test_image/image_raw'),
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
