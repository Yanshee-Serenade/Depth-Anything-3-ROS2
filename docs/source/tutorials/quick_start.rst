Quick Start Tutorial
====================

This tutorial will get you up and running with Depth Anything 3 ROS2 wrapper
in just a few minutes.

Prerequisites
-------------

Before starting, ensure you have:

* Ubuntu 22.04 LTS
* ROS2 Humble installed
* Python 3.10 or higher
* CUDA 12.x (optional, for GPU acceleration)
* A camera publishing ``sensor_msgs/Image`` messages

Installation
------------

Step 1: Clone Repository
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd ~/ros2_ws/src
    git clone https://github.com/yourusername/depth_anything_3_ros2.git
    cd ~/ros2_ws

Step 2: Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Install ROS2 dependencies
    sudo apt update
    sudo apt install ros-humble-cv-bridge \
                     ros-humble-image-transport \
                     ros-humble-vision-opencv

    # Install Python dependencies
    cd ~/ros2_ws/src/depth_anything_3_ros2
    pip3 install -r requirements.txt

Step 3: Build Package
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd ~/ros2_ws
    colcon build --packages-select depth_anything_3_ros2
    source install/setup.bash

Step 4: Download Sample Images (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For testing without a camera:

.. code-block:: bash

    cd ~/ros2_ws/src/depth_anything_3_ros2/examples/scripts
    ./download_samples.sh

Basic Usage
-----------

Using with USB Camera
^^^^^^^^^^^^^^^^^^^^^

**Terminal 1**: Launch camera driver

.. code-block:: bash

    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p image_size:="[640,480]" \
        -p camera_frame_id:=camera_optical_frame

**Terminal 2**: Launch depth estimation

.. code-block:: bash

    source ~/ros2_ws/install/setup.bash
    ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
        model_name:=depth-anything/DA3-BASE \
        device:=cuda \
        input_topic:=/image_raw

**Terminal 3**: Visualize with RViz2

.. code-block:: bash

    rviz2 -d ~/ros2_ws/src/depth_anything_3_ros2/rviz/default.rviz

You should now see:

* Original camera image
* Depth map (grayscale)
* Colored depth visualization
* Confidence map

Using with Sample Images
^^^^^^^^^^^^^^^^^^^^^^^^^

Without ROS2, for quick testing:

.. code-block:: bash

    cd ~/ros2_ws/src/depth_anything_3_ros2/examples/scripts
    python3 test_with_images.py \
        --input ../sample_images/indoor_scene.jpg \
        --output depth_output.png

Using Docker
^^^^^^^^^^^^

For isolated environment:

.. code-block:: bash

    # Build CPU image
    docker compose build cpu

    # Run with USB camera
    docker compose up usb-camera

    # Or use GPU image
    docker compose build gpu
    docker compose up gpu

Understanding the Output
------------------------

Topics Published
^^^^^^^^^^^^^^^^

The node publishes the following topics:

* ``~/depth``: Raw depth map (32-bit float, meters)
* ``~/depth_colored``: Colored visualization (8-bit RGB)
* ``~/confidence``: Confidence scores (32-bit float, 0-1)
* ``~/depth/camera_info``: Camera calibration info

Topic names are in the node's namespace. Use ``ros2 topic list`` to see
full names.

Visualization
^^^^^^^^^^^^^

In RViz2, add displays for:

1. **Image**: Subscribe to ``/depth_anything_3/depth_colored``
2. **Image**: Subscribe to ``/depth_anything_3/confidence``
3. **DepthCloud**: Convert depth to point cloud (optional)

Performance Metrics
^^^^^^^^^^^^^^^^^^^

Check node performance:

.. code-block:: bash

    # Monitor inference rate
    ros2 topic hz /depth_anything_3/depth

    # Expected rates:
    # - DA3-SMALL: 20-30 Hz (GPU) / 2-5 Hz (CPU)
    # - DA3-BASE: 15-25 Hz (GPU) / 1-3 Hz (CPU)
    # - DA3-LARGE: 10-15 Hz (GPU) / 0.5-1 Hz (CPU)

Configuration
-------------

Parameters
^^^^^^^^^^

Key parameters you can adjust:

.. code-block:: yaml

    depth_anything_3_node:
      ros__parameters:
        # Model selection
        model_name: "depth-anything/DA3-BASE"  # SMALL, BASE, or LARGE

        # Device
        device: "cuda"  # cuda or cpu

        # Processing
        publish_colored_depth: true
        publish_confidence: true
        confidence_threshold: 0.5

        # Performance
        queue_size: 1  # Lower for less latency, higher for reliability

Custom Parameters File
^^^^^^^^^^^^^^^^^^^^^^

Create ``my_params.yaml``:

.. code-block:: yaml

    /**:
      ros__parameters:
        model_name: "depth-anything/DA3-SMALL"
        device: "cuda"
        colormap: "viridis"
        confidence_threshold: 0.6

Launch with custom parameters:

.. code-block:: bash

    ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
        params_file:=my_params.yaml

Next Steps
----------

Now that you have the basics working, explore:

* :doc:`usb_camera` - Detailed USB camera configuration
* :doc:`multi_camera` - Using multiple cameras simultaneously
* :doc:`performance_tuning` - Optimizing for your hardware
* :doc:`../api/modules` - Complete API reference

Troubleshooting
---------------

Model Download Fails
^^^^^^^^^^^^^^^^^^^^

**Problem**: ``Connection timeout`` or ``Cannot download model``

**Solution**: Ensure internet connection and Hugging Face access:

.. code-block:: bash

    # Test connection
    ping huggingface.co

    # Pre-download model
    python3 -c "from transformers import AutoImageProcessor; \
                AutoImageProcessor.from_pretrained('depth-anything/DA3-BASE')"

CUDA Out of Memory
^^^^^^^^^^^^^^^^^^

**Problem**: ``CUDA out of memory`` error

**Solutions**:

1. Use smaller model:

   .. code-block:: bash

       model_name:=depth-anything/DA3-SMALL

2. Use CPU mode:

   .. code-block:: bash

       device:=cpu

3. Reduce input resolution (modify camera settings)

No Depth Output
^^^^^^^^^^^^^^^

**Problem**: Node runs but no depth images published

**Check**:

1. Input images arriving:

   .. code-block:: bash

       ros2 topic hz /image_raw

2. Topic remapping correct:

   .. code-block:: bash

       ros2 topic list | grep depth_anything

3. Check node logs:

   .. code-block:: bash

       ros2 node info /depth_anything_3_node

Low Frame Rate
^^^^^^^^^^^^^^

**Problem**: Depth estimation too slow

**Solutions**:

* Use GPU: ``device:=cuda``
* Use smaller model: ``DA3-SMALL``
* Optimize with TensorRT:

  .. code-block:: bash

      cd examples/scripts
      python3 optimize_tensorrt.py \
          --model depth-anything/DA3-BASE \
          --output da3_base_trt.pth

Getting Help
------------

If you encounter issues:

1. Check the logs:

   .. code-block:: bash

       ros2 run depth_anything_3_ros2 depth_anything_3_node --ros-args --log-level debug

2. Search existing issues on GitHub

3. Create a new issue with:

   * Your environment details (OS, ROS version, GPU)
   * Complete error messages
   * Steps to reproduce
