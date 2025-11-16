Multi-Camera Setup Tutorial
============================

This tutorial demonstrates how to use multiple cameras simultaneously with
Depth Anything 3 ROS2 wrapper for applications like:

* 360-degree depth coverage
* Stereo or multi-view depth estimation
* Robot perception from multiple viewpoints
* Synchronized multi-camera recording

Architecture Overview
---------------------

The multi-camera setup uses ROS2 namespaces to isolate each camera's topics:

.. code-block:: text

    /cam_left/
        /image_raw              (from camera driver)
        /camera_info            (from camera driver)
        /depth_anything_3/
            /depth              (depth output)
            /depth_colored      (visualization)
            /confidence         (confidence map)

    /cam_right/
        /image_raw
        /camera_info
        /depth_anything_3/
            /depth
            /depth_colored
            /confidence

Each camera runs in its own namespace, preventing topic conflicts.

Basic Multi-Camera Setup
-------------------------

Method 1: Using Built-in Launch File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to launch multiple cameras:

.. code-block:: bash

    ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
        camera_namespaces:=cam_left,cam_right \
        image_topics:=/cam_left/image_raw,/cam_right/image_raw \
        model_name:=depth-anything/DA3-BASE \
        device:=cuda

This will create two depth estimation nodes, one for each camera.

Method 2: Manual Node Launch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Launch each node individually for more control:

.. code-block:: bash

    # Camera 1 depth estimation
    ros2 run depth_anything_3_ros2 depth_anything_3_node \
        --ros-args \
        -r __ns:=/cam_left \
        -p model_name:=depth-anything/DA3-BASE \
        -p device:=cuda \
        -r ~/image_raw:=/cam_left/image_raw

    # Camera 2 depth estimation
    ros2 run depth_anything_3_ros2 depth_anything_3_node \
        --ros-args \
        -r __ns:=/cam_right \
        -p model_name:=depth-anything/DA3-BASE \
        -p device:=cuda \
        -r ~/image_raw:=/cam_right/image_raw

Dual USB Camera Example
------------------------

Complete setup for two USB cameras:

**Terminal 1**: Left camera driver

.. code-block:: bash

    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -r __ns:=/cam_left \
        -p video_device:=/dev/video0 \
        -p image_size:="[640,480]"

**Terminal 2**: Right camera driver

.. code-block:: bash

    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -r __ns:=/cam_right \
        -p video_device:=/dev/video2 \
        -p image_size:="[640,480]"

**Terminal 3**: Multi-camera depth estimation

.. code-block:: bash

    ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
        camera_namespaces:=cam_left,cam_right \
        image_topics:=/cam_left/image_raw,/cam_right/image_raw

Three-Camera Setup
------------------

For 360-degree or wide-area coverage:

.. code-block:: bash

    # Launch all three cameras
    ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
        camera_namespaces:=cam_left,cam_center,cam_right \
        image_topics:=/cam_left/image_raw,/cam_center/image_raw,/cam_right/image_raw \
        model_name:=depth-anything/DA3-SMALL \
        device:=cuda

Custom Launch File Example
---------------------------

Create a custom launch file for your specific setup:

**File**: ``my_multi_camera.launch.py``

.. code-block:: python

    from launch import LaunchDescription
    from launch_ros.actions import Node
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration

    def generate_launch_description():
        # Declare arguments
        model_arg = DeclareLaunchArgument(
            'model_name',
            default_value='depth-anything/DA3-BASE',
            description='DA3 model to use'
        )

        device_arg = DeclareLaunchArgument(
            'device',
            default_value='cuda',
            description='Device: cuda or cpu'
        )

        # Get configurations
        model_name = LaunchConfiguration('model_name')
        device = LaunchConfiguration('device')

        return LaunchDescription([
            model_arg,
            device_arg,

            # Left camera
            Node(
                package='v4l2_camera',
                executable='v4l2_camera_node',
                namespace='cam_left',
                name='camera',
                parameters=[{
                    'video_device': '/dev/video0',
                    'image_size': [640, 480],
                }],
            ),

            # Right camera
            Node(
                package='v4l2_camera',
                executable='v4l2_camera_node',
                namespace='cam_right',
                name='camera',
                parameters=[{
                    'video_device': '/dev/video2',
                    'image_size': [640, 480],
                }],
            ),

            # Left depth estimation
            Node(
                package='depth_anything_3_ros2',
                executable='depth_anything_3_node',
                namespace='cam_left',
                name='depth_anything_3_node',
                parameters=[{
                    'model_name': model_name,
                    'device': device,
                }],
                remappings=[
                    ('~/image_raw', '/cam_left/image_raw'),
                ],
            ),

            # Right depth estimation
            Node(
                package='depth_anything_3_ros2',
                executable='depth_anything_3_node',
                namespace='cam_right',
                name='depth_anything_3_node',
                parameters=[{
                    'model_name': model_name,
                    'device': device,
                }],
                remappings=[
                    ('~/image_raw', '/cam_right/image_raw'),
                ],
            ),
        ])

Frame Synchronization
---------------------

Synchronize frames from multiple cameras:

Using Python Script
^^^^^^^^^^^^^^^^^^^

The package includes a synchronization utility:

.. code-block:: bash

    # Terminal 1: Launch multi-camera setup
    ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
        camera_namespaces:=cam_left,cam_right \
        image_topics:=/cam_left/image_raw,/cam_right/image_raw

    # Terminal 2: Run synchronizer
    cd examples/scripts
    python3 multi_camera_sync.py \
        --cameras cam_left cam_right \
        --sync-threshold 0.05 \
        --output synchronized_depth/

This will save synchronized depth frames from all cameras.

Using message_filters
^^^^^^^^^^^^^^^^^^^^^^

For custom synchronization in your code:

.. code-block:: python

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from message_filters import ApproximateTimeSynchronizer, Subscriber

    class MultiCameraNode(Node):
        def __init__(self):
            super().__init__('multi_camera_node')

            # Create subscribers
            self.sub_left = Subscriber(
                self, Image, '/cam_left/depth_anything_3/depth'
            )
            self.sub_right = Subscriber(
                self, Image, '/cam_right/depth_anything_3/depth'
            )

            # Synchronize
            self.sync = ApproximateTimeSynchronizer(
                [self.sub_left, self.sub_right],
                queue_size=10,
                slop=0.05  # 50ms tolerance
            )
            self.sync.registerCallback(self.sync_callback)

        def sync_callback(self, left_msg, right_msg):
            # Process synchronized frames
            self.get_logger().info('Received synchronized frames')

Performance Considerations
--------------------------

GPU Memory Management
^^^^^^^^^^^^^^^^^^^^^

Multiple nodes share GPU memory. Monitor usage:

.. code-block:: bash

    # Monitor GPU memory
    nvidia-smi -l 1

If running out of memory:

1. Use smaller model (DA3-SMALL)
2. Reduce camera resolution
3. Use CPU for some cameras:

   .. code-block:: bash

       # GPU for left, CPU for right
       ros2 run depth_anything_3_ros2 depth_anything_3_node \
           --ros-args -r __ns:=/cam_left -p device:=cuda

       ros2 run depth_anything_3_ros2 depth_anything_3_node \
           --ros-args -r __ns:=/cam_right -p device:=cpu

Processing Load Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distribute processing across cores:

.. code-block:: bash

    # Set CPU affinity for each node
    taskset -c 0-3 ros2 run ... # Cameras 1-2 on cores 0-3
    taskset -c 4-7 ros2 run ... # Cameras 3-4 on cores 4-7

Model Loading
^^^^^^^^^^^^^

To save memory, all nodes can share the same model. This happens
automatically when using the same model name.

Applications
------------

Panoramic Depth
^^^^^^^^^^^^^^^

Merge depth from multiple cameras for wide field of view:

.. code-block:: bash

    # Setup: 3 cameras at 120-degree intervals
    # Left: -60 degrees
    # Center: 0 degrees
    # Right: +60 degrees

    ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
        camera_namespaces:=cam_left,cam_center,cam_right \
        image_topics:=/cam_left/image_raw,/cam_center/image_raw,/cam_right/image_raw

Stereo Depth Fusion
^^^^^^^^^^^^^^^^^^^

Combine monocular depth with stereo baseline:

.. code-block:: python

    # Both cameras provide depth
    # Fuse for more accurate 3D reconstruction
    # See examples/scripts/multi_camera_sync.py

Mobile Robot Perception
^^^^^^^^^^^^^^^^^^^^^^^

Multiple cameras for full coverage:

.. code-block:: text

    Front camera: Navigation
    Rear camera: Backup assist
    Side cameras: Obstacle detection

Visualization
-------------

RViz2 Configuration
^^^^^^^^^^^^^^^^^^^

View all cameras in RViz2:

1. Add Image display for each camera:

   * /cam_left/depth_anything_3/depth_colored
   * /cam_right/depth_anything_3/depth_colored

2. Arrange displays in grid layout

3. Save configuration:

   .. code-block:: bash

       File -> Save Config As -> multi_camera.rviz

rqt_image_view
^^^^^^^^^^^^^^

View multiple camera streams:

.. code-block:: bash

    # Terminal 1: Left camera
    rqt_image_view /cam_left/depth_anything_3/depth_colored

    # Terminal 2: Right camera
    rqt_image_view /cam_right/depth_anything_3/depth_colored

Recording and Playback
-----------------------

Record Multi-Camera Data
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Record all topics
    ros2 bag record -a -o multi_camera_recording

    # Or record specific topics
    ros2 bag record \
        /cam_left/depth_anything_3/depth \
        /cam_right/depth_anything_3/depth \
        -o multi_camera_depth

Batch Processing
^^^^^^^^^^^^^^^^

Process recorded bags:

.. code-block:: bash

    cd examples/scripts
    ./ros2_batch_process.sh \
        -i ./input_bags/ \
        -o ./output_bags/ \
        -m depth-anything/DA3-BASE

Troubleshooting
---------------

Cameras Interfering
^^^^^^^^^^^^^^^^^^^

**Problem**: Topics from different cameras mixed up

**Solution**: Ensure proper namespacing:

.. code-block:: bash

    ros2 topic list  # Verify namespace isolation

USB Bandwidth Issues
^^^^^^^^^^^^^^^^^^^^

**Problem**: Cameras dropping frames or not starting

**Solution**:

1. Connect cameras to different USB controllers
2. Reduce resolution/framerate
3. Use USB 3.0 ports

Different Frame Rates
^^^^^^^^^^^^^^^^^^^^^

**Problem**: Cameras running at different rates

**Solution**: Explicitly set framerate:

.. code-block:: bash

    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p framerate:=30.0  # Force 30 FPS

Synchronization Failing
^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Frames not synchronizing properly

**Solutions**:

1. Increase sync threshold:

   .. code-block:: python

       slop=0.1  # 100ms tolerance

2. Use hardware sync if available

3. Check timestamp sources:

   .. code-block:: bash

       ros2 topic echo /cam_left/image_raw --field header.stamp

Next Steps
----------

* :doc:`performance_tuning` - Optimize multi-camera performance
* See ``examples/scripts/multi_camera_sync.py`` for synchronization
* :doc:`../api/modules` - API reference
