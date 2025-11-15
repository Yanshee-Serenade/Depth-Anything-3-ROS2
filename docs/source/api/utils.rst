Utilities Module
================

.. automodule:: depth_anything_3_ros2.utils
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

Depth Processing
^^^^^^^^^^^^^^^^

.. autofunction:: depth_anything_3_ros2.utils.normalize_depth

.. autofunction:: depth_anything_3_ros2.utils.colorize_depth

.. autofunction:: depth_anything_3_ros2.utils.depth_to_meters

Image Processing
^^^^^^^^^^^^^^^^

.. autofunction:: depth_anything_3_ros2.utils.resize_image

Confidence Processing
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: depth_anything_3_ros2.utils.compute_confidence_mask

Classes
-------

PerformanceMetrics
^^^^^^^^^^^^^^^^^^

.. autoclass:: depth_anything_3_ros2.utils.PerformanceMetrics
   :members:
   :special-members: __init__

Usage Examples
--------------

Depth Normalization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from depth_anything_3_ros2.utils import normalize_depth
    import numpy as np

    # Normalize depth map
    depth = np.random.rand(480, 640).astype(np.float32)
    depth_normalized = normalize_depth(depth)

Depth Colorization
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from depth_anything_3_ros2.utils import colorize_depth

    # Colorize with different colormaps
    colored_turbo = colorize_depth(depth, colormap='turbo')
    colored_viridis = colorize_depth(depth, colormap='viridis')

Performance Tracking
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from depth_anything_3_ros2.utils import PerformanceMetrics

    # Create metrics tracker
    metrics = PerformanceMetrics(window_size=30)

    # Update with timing data
    metrics.update(inference_time=0.05, total_time=0.06)

    # Get current metrics
    stats = metrics.get_metrics()
    print(f"FPS: {stats['fps']:.2f}")
    print(f"Avg inference: {stats['avg_inference_ms']:.1f} ms")
