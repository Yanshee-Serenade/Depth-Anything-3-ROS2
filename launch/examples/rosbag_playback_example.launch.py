"""
Example launch file for ROS2 bag playback.

This example shows how to use Depth Anything 3 with pre-recorded ROS2 bags.
Useful for offline processing and testing.

Usage:
    ros2 launch depth_anything_3_ros2 rosbag_playback_example.launch.py \
        bag_file:=/path/to/your/recording.db3 \
        image_topic:=/camera/image_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for rosbag playback example."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_file',
            default_value='',
            description='Path to ROS2 bag file'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Image topic in the bag file'
        ),
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        ),
        DeclareLaunchArgument(
            'playback_rate',
            default_value='1.0',
            description='Playback rate (1.0 = real-time)'
        ),
        DeclareLaunchArgument(
            'record_output',
            default_value='false',
            description='Whether to record depth output to new bag'
        ),

        # Play ROS2 bag
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play',
                LaunchConfiguration('bag_file'),
                '--rate', LaunchConfiguration('playback_rate'),
                '--loop'  # Remove this if you don't want looping
            ],
            output='screen'
        ),

        # Launch Depth Anything 3
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            output='screen',
            remappings=[
                ('~/image_raw', LaunchConfiguration('image_topic')),
            ],
            parameters=[{
                'model_name': LaunchConfiguration('model_name'),
                'device': 'cuda',
                'normalize_depth': True,
                'publish_colored': True,
                'publish_confidence': True,
                'colormap': 'turbo',
                'log_inference_time': True,
                'queue_size': 10,  # Larger queue for bag playback
            }]
        ),

        # Optionally record output (uncomment to enable)
        # ExecuteProcess(
        #     cmd=[
        #         'ros2', 'bag', 'record',
        #         '-o', 'depth_output',
        #         '/depth_anything_3/depth',
        #         '/depth_anything_3/depth_colored',
        #         '/depth_anything_3/confidence'
        #     ],
        #     output='screen',
        #     condition=IfCondition(LaunchConfiguration('record_output'))
        # ),
    ])
