"""
Optimized launch file for high-performance Depth Anything 3 (>30 FPS).

This launch file is optimized for NVIDIA Jetson Orin AGX to achieve >30 FPS
at 1080p with full depth and confidence outputs.

Optimizations:
- TensorRT INT8/FP16 inference
- GPU-accelerated upsampling
- Async colorization
- 384x384 model input (faster inference)
- Subscriber checks (skip work if no subscribers)
- DA3-SMALL model by default (faster)

Usage:
    # Standard optimized mode (PyTorch FP16)
    ros2 launch depth_anything_3_ros2 depth_anything_3_optimized.launch.py

    # TensorRT INT8 mode (fastest, requires converted model)
    ros2 launch depth_anything_3_ros2 depth_anything_3_optimized.launch.py \
        backend:=tensorrt_int8 \
        trt_model_path:=/path/to/da3_small_int8.pth

    # TensorRT FP16 mode (good balance)
    ros2 launch depth_anything_3_ros2 depth_anything_3_optimized.launch.py \
        backend:=tensorrt_fp16 \
        trt_model_path:=/path/to/da3_small_fp16.pth
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate optimized launch description."""

    return LaunchDescription([
        # Camera topic configuration
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Input image topic from camera'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/camera_info',
            description='Input camera info topic'
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace for the node'
        ),

        # Model configuration (optimized defaults)
        DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-SMALL',
            description='Model (DA3-SMALL recommended for speed)'
        ),
        DeclareLaunchArgument(
            'backend',
            default_value='pytorch',
            description='Backend: pytorch, tensorrt_fp16, tensorrt_int8'
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda',
            description='Inference device: cuda or cpu'
        ),
        DeclareLaunchArgument(
            'cache_dir',
            default_value='',
            description='Model cache directory'
        ),
        DeclareLaunchArgument(
            'trt_model_path',
            default_value='',
            description='Path to TensorRT model (required for TensorRT backend)'
        ),

        # Image processing (optimized for >30 FPS)
        DeclareLaunchArgument(
            'model_input_height',
            default_value='384',
            description='Model input height (384 for speed, 518 for quality)'
        ),
        DeclareLaunchArgument(
            'model_input_width',
            default_value='384',
            description='Model input width (384 for speed, 518 for quality)'
        ),
        DeclareLaunchArgument(
            'output_height',
            default_value='1080',
            description='Output depth map height (1080p)'
        ),
        DeclareLaunchArgument(
            'output_width',
            default_value='1920',
            description='Output depth map width (1080p)'
        ),
        DeclareLaunchArgument(
            'input_encoding',
            default_value='bgr8',
            description='Input image encoding'
        ),

        # GPU optimization
        DeclareLaunchArgument(
            'enable_upsampling',
            default_value='true',
            description='Enable GPU upsampling to output resolution'
        ),
        DeclareLaunchArgument(
            'upsample_mode',
            default_value='bilinear',
            description='Upsampling mode: bilinear (fast), bicubic (quality), nearest'
        ),
        DeclareLaunchArgument(
            'use_cuda_streams',
            default_value='false',
            description='Enable CUDA streams for pipeline parallelism (experimental)'
        ),

        # Output configuration
        DeclareLaunchArgument(
            'normalize_depth',
            default_value='true',
            description='Normalize depth to [0, 1] range'
        ),
        DeclareLaunchArgument(
            'publish_colored',
            default_value='true',
            description='Publish colorized depth visualization'
        ),
        DeclareLaunchArgument(
            'publish_confidence',
            default_value='true',
            description='Publish confidence map'
        ),
        DeclareLaunchArgument(
            'colormap',
            default_value='turbo',
            description='Colormap for visualization'
        ),
        DeclareLaunchArgument(
            'async_colorization',
            default_value='true',
            description='Async colorization (off critical path for >30 FPS)'
        ),
        DeclareLaunchArgument(
            'check_subscribers',
            default_value='true',
            description='Skip colorization if no subscribers (optimization)'
        ),

        # Performance parameters
        DeclareLaunchArgument(
            'queue_size',
            default_value='1',
            description='Queue size (1 for latest frame only)'
        ),
        DeclareLaunchArgument(
            'log_inference_time',
            default_value='true',
            description='Log performance metrics every 5 seconds'
        ),

        # Optimized Node
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node_optimized',
            name='depth_anything_3_optimized',
            namespace=LaunchConfiguration('namespace'),
            output='screen',
            remappings=[
                ('~/image_raw', LaunchConfiguration('image_topic')),
                ('~/camera_info', LaunchConfiguration('camera_info_topic')),
            ],
            parameters=[{
                # Model configuration
                'model_name': LaunchConfiguration('model_name'),
                'backend': LaunchConfiguration('backend'),
                'device': LaunchConfiguration('device'),
                'cache_dir': LaunchConfiguration('cache_dir'),
                'trt_model_path': LaunchConfiguration('trt_model_path'),

                # Image processing
                'model_input_height': LaunchConfiguration('model_input_height'),
                'model_input_width': LaunchConfiguration('model_input_width'),
                'output_height': LaunchConfiguration('output_height'),
                'output_width': LaunchConfiguration('output_width'),
                'input_encoding': LaunchConfiguration('input_encoding'),

                # GPU optimization
                'enable_upsampling': LaunchConfiguration('enable_upsampling'),
                'upsample_mode': LaunchConfiguration('upsample_mode'),
                'use_cuda_streams': LaunchConfiguration('use_cuda_streams'),

                # Output configuration
                'normalize_depth': LaunchConfiguration('normalize_depth'),
                'publish_colored': LaunchConfiguration('publish_colored'),
                'publish_confidence': LaunchConfiguration('publish_confidence'),
                'colormap': LaunchConfiguration('colormap'),
                'async_colorization': LaunchConfiguration('async_colorization'),
                'check_subscribers': LaunchConfiguration('check_subscribers'),

                # Performance
                'queue_size': LaunchConfiguration('queue_size'),
                'log_inference_time': LaunchConfiguration('log_inference_time'),
            }]
        ),
    ])
