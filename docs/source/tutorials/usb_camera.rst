USB Camera Setup Tutorial
=========================

This tutorial covers setting up USB cameras with Depth Anything 3 ROS2 wrapper.
The wrapper is completely camera-agnostic and works with any USB camera that
publishes standard ROS2 image messages.

Supported USB Cameras
----------------------

This package works with ANY USB camera, including:

* Generic USB webcams (Logitech, Microsoft, etc.)
* Industrial USB cameras (FLIR, Basler, etc.)
* USB endoscopes and inspection cameras
* Laptop built-in webcams
* Any camera compatible with v4l2 or other ROS2 drivers

Camera Driver Selection
------------------------

Option 1: v4l2_camera (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Best for standard USB cameras using Video4Linux2.

**Install**:

.. code-block:: bash

    sudo apt install ros-humble-v4l2-camera

**Launch**:

.. code-block:: bash

    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p video_device:=/dev/video0 \
        -p image_size:="[640,480]" \
        -p camera_frame_id:=camera_optical_frame \
        -p pixel_format:=YUYV

Option 2: usb_cam
^^^^^^^^^^^^^^^^^

Alternative USB camera driver.

**Install**:

.. code-block:: bash

    sudo apt install ros-humble-usb-cam

**Launch**:

.. code-block:: bash

    ros2 run usb_cam usb_cam_node_exe \
        --ros-args \
        -p video_device:=/dev/video0 \
        -p image_width:=640 \
        -p image_height:=480 \
        -p camera_name:=usb_camera

Option 3: Industrial Camera Drivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For industrial cameras, use manufacturer-specific ROS2 packages:

* **FLIR/Spinnaker**: spinnaker_camera_driver
* **Basler**: pylon_ros2_camera_driver
* **IDS**: ids_peak_ros2

Finding Your Camera
-------------------

List Connected Cameras
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # List v4l2 devices
    v4l2-ctl --list-devices

    # Output example:
    # USB Camera (usb-0000:00:14.0-1):
    #     /dev/video0
    #     /dev/video1

    # Check camera capabilities
    v4l2-ctl --device=/dev/video0 --all

Identify Camera Device Number
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # List all video devices
    ls -lh /dev/video*

    # Test camera with simple tool
    sudo apt install cheese
    cheese  # Opens camera viewer

Camera Permissions
^^^^^^^^^^^^^^^^^^

If you get permission denied:

.. code-block:: bash

    # Add user to video group
    sudo usermod -a -G video $USER

    # Log out and back in, then verify
    groups | grep video

    # Or set permissions directly (temporary)
    sudo chmod 666 /dev/video0

Complete Setup Example
-----------------------

Example 1: Logitech C920 Webcam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Step 1**: Verify camera:

.. code-block:: bash

    v4l2-ctl --device=/dev/video0 --list-formats-ext

**Step 2**: Create launch file ``logitech_c920.launch.py``:

.. code-block:: python

    from launch import LaunchDescription
    from launch_ros.actions import Node

    def generate_launch_description():
        return LaunchDescription([
            # Camera driver
            Node(
                package='v4l2_camera',
                executable='v4l2_camera_node',
                name='camera',
                parameters=[{
                    'video_device': '/dev/video0',
                    'image_size': [1920, 1080],
                    'camera_frame_id': 'camera_optical_frame',
                    'pixel_format': 'MJPEG',
                }],
            ),

            # Depth estimation
            Node(
                package='depth_anything_3_ros2',
                executable='depth_anything_3_node',
                name='depth_anything_3_node',
                parameters=[{
                    'model_name': 'depth-anything/DA3-BASE',
                    'device': 'cuda',
                }],
                remappings=[
                    ('~/image_raw', '/image_raw'),
                    ('~/camera_info', '/camera_info'),
                ],
            ),
        ])

**Step 3**: Launch:

.. code-block:: bash

    ros2 launch logitech_c920.launch.py

Example 2: Low-Resolution Webcam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For budget webcams with limited resolution:

.. code-block:: bash

    # Terminal 1: Camera at 320x240
    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p video_device:=/dev/video0 \
        -p image_size:="[320,240]" \
        -p camera_frame_id:=camera_optical_frame

    # Terminal 2: Use SMALL model for better performance
    ros2 run depth_anything_3_ros2 depth_anything_3_node \
        --ros-args \
        -p model_name:=depth-anything/DA3-SMALL \
        -p device:=cuda \
        -r ~/image_raw:=/image_raw

Example 3: Multiple USB Cameras
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using multiple cameras simultaneously:

.. code-block:: bash

    # Camera 1 on /dev/video0
    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -r __ns:=/cam1 \
        -p video_device:=/dev/video0 \
        -p image_size:="[640,480]"

    # Camera 2 on /dev/video2
    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -r __ns:=/cam2 \
        -p video_device:=/dev/video2 \
        -p image_size:="[640,480]"

    # Depth estimation for both
    ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
        camera_namespaces:=cam1,cam2 \
        image_topics:=/cam1/image_raw,/cam2/image_raw

Camera Calibration
------------------

For accurate depth estimation, calibrate your camera:

**Step 1**: Install calibration tools:

.. code-block:: bash

    sudo apt install ros-humble-camera-calibration

**Step 2**: Print calibration pattern:

Download and print a checkerboard pattern from:
https://github.com/opencv/opencv/blob/master/doc/pattern.png

**Step 3**: Run calibration:

.. code-block:: bash

    ros2 run camera_calibration cameracalibrator \
        --size 8x6 \
        --square 0.024 \
        image:=/image_raw

**Step 4**: Move checkerboard around until X, Y, Size, Skew bars are green

**Step 5**: Click "Calibrate", then "Save"

**Step 6**: Copy calibration file:

.. code-block:: bash

    # Calibration saved to ~/.ros/camera_info/
    # Copy to your config directory
    cp ~/.ros/camera_info/camera.yaml \
       ~/ros2_ws/src/depth_anything_3_ros2/config/my_camera.yaml

**Step 7**: Use calibration:

.. code-block:: bash

    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p camera_info_url:=file:///path/to/my_camera.yaml

Performance Optimization
------------------------

Adjust Resolution
^^^^^^^^^^^^^^^^^

Higher resolution = better quality but slower:

.. code-block:: bash

    # Fast (320x240)
    -p image_size:="[320,240]"

    # Balanced (640x480)
    -p image_size:="[640,480]"

    # High quality (1280x720)
    -p image_size:="[1280,720]"

Choose Pixel Format
^^^^^^^^^^^^^^^^^^^

Some formats are more efficient:

.. code-block:: bash

    # MJPEG: Lower CPU usage, compressed
    -p pixel_format:=MJPEG

    # YUYV: Higher quality, more CPU
    -p pixel_format:=YUYV

    # RGB: Best quality, highest CPU
    -p pixel_format:=RGB3

Adjust Frame Rate
^^^^^^^^^^^^^^^^^

Limit camera FPS to match processing speed:

.. code-block:: bash

    # 15 FPS
    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p framerate:=15.0

    # 30 FPS
    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p framerate:=30.0

Troubleshooting
---------------

Camera Not Detected
^^^^^^^^^^^^^^^^^^^

**Check USB connection**:

.. code-block:: bash

    lsusb  # Should show your camera
    dmesg | grep video  # Check kernel messages

**Try different USB port**: Some cameras need USB 3.0

**Check device permissions**: See "Camera Permissions" section above

Poor Image Quality
^^^^^^^^^^^^^^^^^^

**Adjust exposure**:

.. code-block:: bash

    v4l2-ctl --device=/dev/video0 \
        --set-ctrl=exposure_auto=1 \
        --set-ctrl=exposure_absolute=150

**Adjust brightness/contrast**:

.. code-block:: bash

    v4l2-ctl --device=/dev/video0 \
        --set-ctrl=brightness=128 \
        --set-ctrl=contrast=32

Low Frame Rate
^^^^^^^^^^^^^^

**Solutions**:

1. Reduce resolution
2. Use MJPEG format
3. Use DA3-SMALL model
4. Enable GPU with ``device:=cuda``
5. Limit camera FPS to match processing

Camera Disconnects
^^^^^^^^^^^^^^^^^^

**Power issue**: Use powered USB hub

**USB bandwidth**: Don't use multiple high-res cameras on same USB controller

**Driver issue**: Try different driver (v4l2_camera vs usb_cam)

Example Launch Files
--------------------

All-in-One Launch
^^^^^^^^^^^^^^^^^

Complete launch file with RViz:

.. code-block:: python

    from launch import LaunchDescription
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    import os

    def generate_launch_description():
        pkg_dir = get_package_share_directory('depth_anything_3_ros2')
        rviz_config = os.path.join(pkg_dir, 'rviz', 'default.rviz')

        return LaunchDescription([
            # USB Camera
            Node(
                package='v4l2_camera',
                executable='v4l2_camera_node',
                name='usb_camera',
                parameters=[{
                    'video_device': '/dev/video0',
                    'image_size': [640, 480],
                    'pixel_format': 'MJPEG',
                    'framerate': 30.0,
                }],
            ),

            # Depth Estimation
            Node(
                package='depth_anything_3_ros2',
                executable='depth_anything_3_node',
                name='depth_anything_3_node',
                parameters=[{
                    'model_name': 'depth-anything/DA3-BASE',
                    'device': 'cuda',
                    'publish_colored_depth': True,
                }],
                remappings=[
                    ('~/image_raw', '/image_raw'),
                ],
            ),

            # RViz2
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
            ),
        ])

Next Steps
----------

* :doc:`multi_camera` - Setup multiple cameras
* :doc:`performance_tuning` - Optimize performance
* :doc:`../examples` - More example configurations
