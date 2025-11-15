"""
Example launch file for generic USB camera using v4l2_camera.

This example shows how to use Depth Anything 3 with a standard USB camera.
Users must have v4l2_camera installed separately:
    sudo apt install ros-humble-v4l2-camera

This file provides ONLY topic remapping - no camera-specific code.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for USB camera example."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='Video device path'
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='640',
            description='Camera image width'
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='480',
            description='Camera image height'
        ),
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        ),

        # Launch v4l2_camera node (user must have this installed)
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera',
            namespace='camera',
            parameters=[{
                'video_device': LaunchConfiguration('video_device'),
                'image_size': [
                    LaunchConfiguration('image_width'),
                    LaunchConfiguration('image_height')
                ],
                'camera_frame_id': 'camera_optical_frame',
            }]
        ),

        # Launch Depth Anything 3 node with topic remapping
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            namespace='camera',
            output='screen',
            remappings=[
                ('~/image_raw', '/camera/image_raw'),
                ('~/camera_info', '/camera/camera_info'),
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
