"""
Example launch file for Intel RealSense camera integration.

This example shows how to connect Depth Anything 3 to RealSense camera topics.
Users must have realsense-ros installed separately:
    sudo apt install ros-humble-realsense2-camera

This file provides ONLY topic remapping - no camera-specific code.
The RealSense camera driver must be launched separately or included here.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for RealSense camera example."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        ),
        DeclareLaunchArgument(
            'launch_realsense',
            default_value='false',
            description='Whether to launch RealSense driver (requires realsense2-camera)'
        ),

        # Optionally launch RealSense camera (uncomment if realsense2-camera is installed)
        # Note: This is commented by default as it requires separate installation
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         PathJoinSubstitution([
        #             FindPackageShare('realsense2_camera'),
        #             'launch',
        #             'rs_launch.py'
        #         ])
        #     ]),
        #     condition=IfCondition(LaunchConfiguration('launch_realsense'))
        # ),

        # Launch Depth Anything 3 with RealSense topic remapping
        # Default RealSense topics: /camera/camera/color/image_raw
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            namespace='camera',
            output='screen',
            remappings=[
                ('~/image_raw', '/camera/camera/color/image_raw'),
                ('~/camera_info', '/camera/camera/color/camera_info'),
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
