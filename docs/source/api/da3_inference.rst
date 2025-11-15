DA3 Inference Module
====================

.. automodule:: depth_anything_3_ros2.da3_inference
   :members:
   :undoc-members:
   :show-inheritance:

Class Reference
---------------

DA3InferenceWrapper
^^^^^^^^^^^^^^^^^^^

.. autoclass:: depth_anything_3_ros2.da3_inference.DA3InferenceWrapper
   :members:
   :special-members: __init__
   :private-members:

Usage Examples
--------------

Basic Inference
^^^^^^^^^^^^^^^

.. code-block:: python

    from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper
    import cv2

    # Initialize wrapper
    model = DA3InferenceWrapper(
        model_name="depth-anything/DA3-BASE",
        device="cuda"
    )

    # Load and process image
    image = cv2.imread("test.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    result = model.inference(image_rgb, return_confidence=True)

    # Access results
    depth = result['depth']
    confidence = result['confidence']

GPU Memory Monitoring
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Check GPU memory usage
    mem_usage = model.get_gpu_memory_usage()
    if mem_usage:
        print(f"Allocated: {mem_usage['allocated_mb']:.1f} MB")
        print(f"Reserved: {mem_usage['reserved_mb']:.1f} MB")

    # Clear cache if needed
    model.clear_cache()

Error Handling
^^^^^^^^^^^^^^

.. code-block:: python

    try:
        result = model.inference(image)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # Handle OOM error
            model.clear_cache()
            # Retry or switch to CPU
        else:
            raise

Device Management
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # CUDA device (with automatic fallback)
    model_gpu = DA3InferenceWrapper(
        model_name="depth-anything/DA3-BASE",
        device="cuda"
    )

    # CPU-only mode
    model_cpu = DA3InferenceWrapper(
        model_name="depth-anything/DA3-BASE",
        device="cpu"
    )
