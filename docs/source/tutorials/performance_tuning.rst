Performance Tuning Guide
=========================

This guide covers optimizing Depth Anything 3 ROS2 wrapper for maximum
performance on various hardware platforms.

Performance Overview
--------------------

Typical Performance
^^^^^^^^^^^^^^^^^^^

On NVIDIA Jetson Orin AGX 64GB:

============= ======= =============== ===============
Model         Device  Input Size      FPS
============= ======= =============== ===============
DA3-SMALL     CUDA    518x518         25-35
DA3-BASE      CUDA    518x518         15-25
DA3-LARGE     CUDA    518x518         8-15
DA3-SMALL     CPU     518x518         3-5
DA3-BASE      CPU     518x518         1-3
DA3-LARGE     CPU     518x518         0.5-1
============= ======= =============== ===============

On Desktop GPU (RTX 3090):

============= ======= =============== ===============
Model         Device  Input Size      FPS
============= ======= =============== ===============
DA3-SMALL     CUDA    518x518         60-80
DA3-BASE      CUDA    518x518         40-60
DA3-LARGE     CUDA    518x518         20-30
============= ======= =============== ===============

Hardware Optimization
---------------------

GPU Acceleration
^^^^^^^^^^^^^^^^

**Enable CUDA**:

.. code-block:: bash

    ros2 run depth_anything_3_ros2 depth_anything_3_node \
        --ros-args -p device:=cuda

**Verify GPU usage**:

.. code-block:: bash

    # Monitor GPU utilization
    nvidia-smi -l 1

    # Check CUDA availability
    python3 -c "import torch; print(torch.cuda.is_available())"

**Optimize GPU memory**:

.. code-block:: yaml

    depth_anything_3_node:
      ros__parameters:
        device: "cuda"
        # Use mixed precision for memory savings
        use_fp16: true  # If supported

TensorRT Optimization (Jetson)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For maximum performance on Jetson platforms:

**Step 1**: Install torch2trt:

.. code-block:: bash

    pip3 install torch2trt

**Step 2**: Optimize model:

.. code-block:: bash

    cd examples/scripts
    python3 optimize_tensorrt.py \
        --model depth-anything/DA3-BASE \
        --output da3_base_trt.pth \
        --precision fp16 \
        --benchmark

**Expected speedup**: 2-3x faster inference

**Step 3**: Modify node to use TRT model (custom integration required)

Model Selection
^^^^^^^^^^^^^^^

Choose the right model for your hardware:

**DA3-SMALL**:

* Best for: Real-time applications, embedded systems
* Speed: Fastest
* Quality: Good
* Use when: FPS > 20 required

**DA3-BASE**:

* Best for: Balanced performance and quality
* Speed: Medium
* Quality: Better
* Use when: General purpose, adequate GPU

**DA3-LARGE**:

* Best for: Offline processing, maximum quality
* Speed: Slowest
* Quality: Best
* Use when: Quality is critical, speed not important

Software Optimization
---------------------

Queue Size Tuning
^^^^^^^^^^^^^^^^^

Adjust message queue for your needs:

.. code-block:: yaml

    depth_anything_3_node:
      ros__parameters:
        # Low latency (drop frames if processing slow)
        queue_size: 1

        # Reliability (buffer more frames, higher latency)
        queue_size: 10

Disable Unused Outputs
^^^^^^^^^^^^^^^^^^^^^^^

Save processing time by disabling unneeded outputs:

.. code-block:: yaml

    depth_anything_3_node:
      ros__parameters:
        # Disable colored depth if not visualizing
        publish_colored_depth: false

        # Disable confidence if not needed
        publish_confidence: false

        # Disable camera parameters if not needed
        publish_camera_params: false

Threading and Concurrency
^^^^^^^^^^^^^^^^^^^^^^^^^^

ROS2 uses multi-threaded executor by default. Optimize with:

.. code-block:: python

    # In custom node
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()

Image Processing Optimization
------------------------------

Input Resolution
^^^^^^^^^^^^^^^^

Lower resolution = faster processing:

.. code-block:: bash

    # Camera settings
    ros2 run v4l2_camera v4l2_camera_node \
        --ros-args \
        -p image_size:="[320,240]"  # Very fast
        # -p image_size:="[640,480]"  # Balanced
        # -p image_size:="[1280,720]"  # High quality

The model will resize images internally, but starting with lower
resolution reduces data transfer and preprocessing overhead.

Image Encoding
^^^^^^^^^^^^^^

Use efficient encodings:

.. code-block:: python

    # In camera driver, prefer:
    # - MJPEG: Compressed, lower bandwidth
    # - RGB8: Direct, no conversion
    # Avoid:
    # - YUV formats: Require conversion

Colormap Selection
^^^^^^^^^^^^^^^^^^

Different colormaps have different performance:

.. code-block:: yaml

    depth_anything_3_node:
      ros__parameters:
        # Fast colormaps
        colormap: "turbo"    # Fastest
        # colormap: "viridis"  # Fast
        # colormap: "jet"      # Medium

Batch Processing Optimization
------------------------------

For offline processing, use batch mode:

**Batch image processing**:

.. code-block:: bash

    cd examples/scripts
    python3 test_with_images.py \
        --input image_dir/ \
        --output depth_dir/ \
        --batch-size 4  # Process 4 images at once

**Batch ROS bag processing**:

.. code-block:: bash

    ./ros2_batch_process.sh \
        -i raw_bags/ \
        -o depth_bags/ \
        -m depth-anything/DA3-SMALL

Network and Communication
--------------------------

QoS Settings
^^^^^^^^^^^^

Optimize Quality of Service for your use case:

.. code-block:: python

    from rclpy.qos import QoSProfile, \
        QoSReliabilityPolicy, QoSHistoryPolicy

    # Low latency
    qos_profile = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1
    )

    # Reliability
    qos_profile = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10
    )

Topic Compression
^^^^^^^^^^^^^^^^^

For network transport:

.. code-block:: bash

    # Install compression
    sudo apt install ros-humble-compressed-image-transport

    # Use compressed topics
    ros2 run image_transport republish \
        raw in:=/image_raw \
        compressed out:=/image_raw/compressed

Advanced Optimizations
----------------------

INT8 Quantization
^^^^^^^^^^^^^^^^^

Reduce model size and increase speed with quantization:

.. code-block:: bash

    cd examples/scripts
    python3 performance_tuning.py quantize \
        --model depth-anything/DA3-BASE \
        --output da3_base_int8.pth \
        --mode dynamic

**Benefits**:

* 50-75% smaller model size
* 20-40% faster inference
* Minimal accuracy loss

**Use when**: Memory or speed constrained

ONNX Export
^^^^^^^^^^^

Export to ONNX for deployment flexibility:

.. code-block:: bash

    python3 performance_tuning.py export-onnx \
        --model depth-anything/DA3-BASE \
        --output da3_base.onnx \
        --benchmark

**Benefits**:

* Cross-platform deployment
* Potential optimizations
* Integration with TensorRT

**Use when**: Deploying to specialized hardware

Model Pruning
^^^^^^^^^^^^^

Remove redundant weights (requires custom implementation):

.. code-block:: python

    import torch.nn.utils.prune as prune

    # Prune 30% of weights
    prune.l1_unstructured(model.layer, 'weight', amount=0.3)

Multi-Camera Optimization
--------------------------

When running multiple cameras:

**Share model weights**:

All nodes automatically share model when using same model name,
saving GPU memory.

**Distribute across GPUs** (if available):

.. code-block:: bash

    # Camera 1 on GPU 0
    CUDA_VISIBLE_DEVICES=0 ros2 run depth_anything_3_ros2 \
        depth_anything_3_node --ros-args -r __ns:=/cam1

    # Camera 2 on GPU 1
    CUDA_VISIBLE_DEVICES=1 ros2 run depth_anything_3_ros2 \
        depth_anything_3_node --ros-args -r __ns:=/cam2

**Stagger processing**:

.. code-block:: python

    # Offset camera start times to avoid processing spikes
    # In launch file, add delays between node starts

Profiling and Benchmarking
---------------------------

Node Profiling
^^^^^^^^^^^^^^

Profile node performance:

.. code-block:: bash

    cd examples/scripts
    python3 profile_node.py \
        --model depth-anything/DA3-BASE \
        --device cuda \
        --duration 60

This outputs:

* Average FPS
* Inference time
* Total processing time
* CPU and memory usage

Layer Profiling
^^^^^^^^^^^^^^^

Identify bottleneck layers:

.. code-block:: bash

    python3 performance_tuning.py profile \
        --model depth-anything/DA3-BASE \
        --layers \
        --iterations 100

Memory Profiling
^^^^^^^^^^^^^^^^

Check memory usage:

.. code-block:: bash

    python3 performance_tuning.py profile \
        --model depth-anything/DA3-BASE \
        --memory

Benchmark Script
^^^^^^^^^^^^^^^^

Comprehensive benchmarking:

.. code-block:: bash

    cd examples/scripts
    python3 benchmark.py \
        --models depth-anything/DA3-SMALL depth-anything/DA3-BASE \
        --devices cuda cpu \
        --image-sizes 320x240 640x480 \
        --iterations 100 \
        --output benchmark_results.json

Platform-Specific Tips
-----------------------

NVIDIA Jetson
^^^^^^^^^^^^^

**Maximize performance**:

.. code-block:: bash

    # Set power mode to maximum
    sudo nvpmodel -m 0
    sudo jetson_clocks

    # Use TensorRT optimization
    cd examples/scripts
    python3 optimize_tensorrt.py \
        --model depth-anything/DA3-SMALL \
        --precision fp16

**Monitor thermals**:

.. code-block:: bash

    tegrastats  # Watch temperature and throttling

Desktop GPU
^^^^^^^^^^^

**Use latest CUDA**:

.. code-block:: bash

    # Check CUDA version
    nvcc --version

    # Install PyTorch with matching CUDA
    pip3 install torch --index-url \
        https://download.pytorch.org/whl/cu121

CPU-Only Systems
^^^^^^^^^^^^^^^^

**Optimize for CPU**:

.. code-block:: bash

    # Use SMALL model
    model_name:=depth-anything/DA3-SMALL

    # Enable OpenMP threading
    export OMP_NUM_THREADS=8

    # Lower resolution
    camera image_size:=[320,240]

Troubleshooting Performance
----------------------------

Low FPS
^^^^^^^

**Diagnosis**:

.. code-block:: bash

    # Check topic rate
    ros2 topic hz /depth_anything_3/depth

    # Check node CPU usage
    top -p $(pgrep -f depth_anything_3_node)

**Solutions**:

1. Use GPU: ``device:=cuda``
2. Smaller model: ``DA3-SMALL``
3. Lower resolution
4. Disable unused outputs
5. TensorRT optimization

High Latency
^^^^^^^^^^^^

**Diagnosis**:

.. code-block:: bash

    ros2 topic delay /depth_anything_3/depth

**Solutions**:

1. Reduce queue_size to 1
2. Use BEST_EFFORT QoS
3. Disable colored depth
4. Process on faster hardware

GPU Out of Memory
^^^^^^^^^^^^^^^^^

**Solutions**:

1. Use smaller model
2. Reduce batch size to 1
3. Lower input resolution
4. Close other GPU applications
5. Use CPU for some cameras in multi-camera setup

Monitoring Tools
----------------

Real-Time Monitoring
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # GPU monitoring
    nvidia-smi -l 1

    # Topic monitoring
    ros2 topic hz /depth_anything_3/depth

    # CPU monitoring
    htop

    # ROS2 node info
    ros2 node info /depth_anything_3_node

Performance Logging
^^^^^^^^^^^^^^^^^^^

Enable detailed logging:

.. code-block:: bash

    ros2 run depth_anything_3_ros2 depth_anything_3_node \
        --ros-args \
        --log-level debug \
        2>&1 | tee performance.log

Best Practices
--------------

1. **Start with defaults**, then optimize
2. **Profile before optimizing** to identify bottlenecks
3. **Test on target hardware** early
4. **Monitor GPU memory** in multi-camera setups
5. **Use TensorRT** on Jetson platforms
6. **Disable unused features** (colored depth, confidence)
7. **Match camera FPS** to processing capability
8. **Use appropriate model** for hardware (SMALL for embedded)

Next Steps
----------

* Try the benchmark script: ``examples/scripts/benchmark.py``
* Profile your setup: ``examples/scripts/profile_node.py``
* Optimize with TensorRT: ``examples/scripts/optimize_tensorrt.py``
* :doc:`../api/modules` - API reference
