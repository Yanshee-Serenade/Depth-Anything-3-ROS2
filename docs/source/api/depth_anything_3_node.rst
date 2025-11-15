ROS2 Node Module
================

.. automodule:: depth_anything_3_ros2.depth_anything_3_node
   :members:
   :undoc-members:
   :show-inheritance:

Class Reference
---------------

DepthAnything3Node
^^^^^^^^^^^^^^^^^^

.. autoclass:: depth_anything_3_ros2.depth_anything_3_node.DepthAnything3Node
   :members:
   :special-members: __init__
   :private-members:

Topics
------

Subscribed Topics
^^^^^^^^^^^^^^^^^

* ``~/image_raw`` (sensor_msgs/Image): Input RGB image from camera
* ``~/camera_info`` (sensor_msgs/CameraInfo): Optional camera intrinsics

Published Topics
^^^^^^^^^^^^^^^^

* ``~/depth`` (sensor_msgs/Image): Depth map (32FC1 encoding)
* ``~/depth_colored`` (sensor_msgs/Image): Colorized depth visualization (BGR8)
* ``~/confidence`` (sensor_msgs/Image): Confidence map (32FC1)
* ``~/depth/camera_info`` (sensor_msgs/CameraInfo): Camera info for depth image

Parameters
----------

Model Configuration
^^^^^^^^^^^^^^^^^^^

* ``model_name`` (string, default: "depth-anything/DA3-BASE")
    Hugging Face model ID or local path

* ``device`` (string, default: "cuda")
    Inference device: cuda or cpu

* ``cache_dir`` (string, default: "")
    Model cache directory (empty for default)

Image Processing
^^^^^^^^^^^^^^^^

* ``inference_height`` (int, default: 518)
    Height for inference (model input size)

* ``inference_width`` (int, default: 518)
    Width for inference (model input size)

* ``input_encoding`` (string, default: "bgr8")
    Expected input encoding (bgr8 or rgb8)

Output Configuration
^^^^^^^^^^^^^^^^^^^^

* ``normalize_depth`` (bool, default: true)
    Normalize depth to [0, 1] range

* ``publish_colored`` (bool, default: true)
    Publish colorized depth visualization

* ``publish_confidence`` (bool, default: true)
    Publish confidence map

* ``colormap`` (string, default: "turbo")
    Colormap for visualization

Performance
^^^^^^^^^^^

* ``queue_size`` (int, default: 1)
    Subscriber queue size (1 for latest only)

* ``processing_threads`` (int, default: 1)
    Number of processing threads

* ``log_inference_time`` (bool, default: false)
    Log per-frame inference time

Usage Example
-------------

.. code-block:: bash

    # Run node with custom parameters
    ros2 run depth_anything_3_ros2 depth_anything_3_node --ros-args \
        -p model_name:=depth-anything/DA3-LARGE \
        -p device:=cuda \
        -p log_inference_time:=true \
        -r ~/image_raw:=/camera/image_raw
