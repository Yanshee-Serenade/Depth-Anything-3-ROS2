API Reference
=============

This section contains the complete API documentation for the Depth Anything 3 ROS2 Wrapper.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   da3_inference
   depth_anything_3_node
   utils

Module Overview
---------------

The package consists of three main Python modules:

1. **da3_inference**: Handles DA3 model loading and inference
2. **depth_anything_3_node**: ROS2 node implementation
3. **utils**: Utility functions for depth processing and visualization

Package Structure
-----------------

.. code-block:: text

    depth_anything_3_ros2/
    ├── __init__.py
    ├── da3_inference.py       # Model inference wrapper
    ├── depth_anything_3_node.py # ROS2 node
    └── utils.py               # Utilities

Quick API Example
-----------------

Here's a quick example of using the inference wrapper:

.. code-block:: python

    from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper
    import numpy as np

    # Initialize model
    model = DA3InferenceWrapper(
        model_name="depth-anything/DA3-BASE",
        device="cuda"
    )

    # Run inference
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = model.inference(image, return_confidence=True)

    # Access results
    depth_map = result['depth']
    confidence = result['confidence']

For detailed API documentation, see the individual module pages.
