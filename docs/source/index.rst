Depth Anything 3 ROS2 Wrapper Documentation
===========================================

Welcome to the comprehensive documentation for the Depth Anything 3 ROS2 Wrapper, a production-ready, camera-agnostic ROS2 package for state-of-the-art monocular depth estimation.

.. image:: https://img.shields.io/badge/ROS2-Humble-blue
   :target: https://docs.ros.org/en/humble/
   :alt: ROS2 Humble

.. image:: https://img.shields.io/badge/Python-3.10+-green
   :target: https://www.python.org/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/CUDA-12.x-orange
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: CUDA 12.x

Overview
--------

This package provides a complete ROS2 integration for `Depth Anything 3 <https://github.com/ByteDance-Seed/Depth-Anything-3>`_,
the latest advancement in monocular depth estimation from ByteDance. It is designed to work seamlessly with any camera
publishing standard ``sensor_msgs/Image`` messages, making it highly versatile for robotics applications.

Key Features
------------

* **Camera-Agnostic Design**: Works with ANY camera publishing standard ROS2 image topics
* **Multiple Model Support**: All DA3 variants (Small, Base, Large, Giant, Nested)
* **CUDA Acceleration**: Optimized for NVIDIA GPUs with automatic CPU fallback
* **Multi-Camera Support**: Run multiple instances for multi-camera setups
* **Real-Time Performance**: Optimized for low latency on Jetson Orin AGX
* **Production Ready**: Comprehensive error handling, logging, and testing
* **Docker Support**: Pre-configured Docker images for easy deployment
* **Complete Documentation**: Extensive guides, tutorials, and API reference

Quick Links
-----------

* :doc:`Installation Guide <guides/installation>`
* :doc:`Quick Start Tutorial <tutorials/quick_start>`
* :doc:`API Reference <api/modules>`
* :doc:`Docker Deployment <guides/docker>`
* :doc:`Performance Benchmarks <guides/performance>`

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   guides/installation
   tutorials/quick_start
   guides/docker
   tutorials/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: User Guides
   :hidden:

   guides/configuration
   guides/camera_integration
   guides/multi_camera
   guides/performance
   guides/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/quick_start
   tutorials/usb_camera
   tutorials/multi_camera
   tutorials/performance_tuning

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/modules
   api/da3_inference
   api/depth_anything_3_node
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   guides/contributing
   guides/testing
   guides/ci_cd

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   changelog
   license
   citation

Acknowledgments
---------------

This package would not be possible without the excellent work of:

* **Depth Anything 3**: ByteDance Seed Team - `GitHub <https://github.com/ByteDance-Seed/Depth-Anything-3>`_ | `Paper <https://arxiv.org/abs/2511.10647>`_
* **ROS2 Community**: For the excellent robotics middleware
* **PyTorch Team**: For the deep learning framework

Support
-------

* **Issues**: `GitHub Issues <https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper/issues>`_
* **Discussions**: `GitHub Discussions <https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper/discussions>`_
* **Documentation**: This site
* **ROS2 Docs**: `ROS2 Humble <https://docs.ros.org/en/humble/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
