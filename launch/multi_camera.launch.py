"""
Multi-camera launch file for Depth Anything 3 ROS2.

This launch file demonstrates running multiple depth estimation nodes
simultaneously for multi-camera setups with proper namespace isolation.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """
    Setup multiple depth estimation nodes based on configuration.

    Args:
        context: Launch context

    Returns:
        List of nodes to launch
    """
    # Get camera configuration
    camera_namespaces = LaunchConfiguration('camera_namespaces').perform(context)
    image_topics = LaunchConfiguration('image_topics').perform(context)
    model_name = LaunchConfiguration('model_name').perform(context)
    device = LaunchConfiguration('device').perform(context)

    # Parse comma-separated lists
    namespaces = [ns.strip() for ns in camera_namespaces.split(',')]
    topics = [topic.strip() for topic in image_topics.split(',')]

    if len(namespaces) != len(topics):
        raise ValueError(
            f"Number of namespaces ({len(namespaces)}) must match "
            f"number of image topics ({len(topics)})"
        )

    nodes = []
    for namespace, image_topic in zip(namespaces, topics):
        node = Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name='depth_anything_3',
            namespace=namespace,
            output='screen',
            remappings=[
                ('~/image_raw', image_topic),
                ('~/camera_info', image_topic.replace('image_raw', 'camera_info')),
            ],
            parameters=[{
                'model_name': model_name,
                'device': device,
                'normalize_depth': True,
                'publish_colored': True,
                'publish_confidence': True,
                'colormap': 'turbo',
                'queue_size': 1,
                'log_inference_time': True,
            }]
        )
        nodes.append(node)

    return nodes


def generate_launch_description():
    """Generate launch description for multi-camera setup."""

    return LaunchDescription([
        # Camera configuration
        DeclareLaunchArgument(
            'camera_namespaces',
            default_value='cam1,cam2,cam3,cam4',
            description='Comma-separated list of camera namespaces'
        ),
        DeclareLaunchArgument(
            'image_topics',
            default_value='/cam1/image_raw,/cam2/image_raw,/cam3/image_raw,/cam4/image_raw',
            description='Comma-separated list of image topics'
        ),

        # Model configuration
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='Hugging Face model ID for all cameras'
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda',
            description='Inference device: cuda or cpu'
        ),

        # Setup nodes with OpaqueFunction
        OpaqueFunction(function=launch_setup),
    ])
