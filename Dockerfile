# Depth Anything 3 ROS2 Wrapper - Docker Image
# Multi-stage build for optimized image size
# Supports both CPU and CUDA builds

# Build arguments
ARG ROS_DISTRO=humble
ARG CUDA_VERSION=12.2.0
ARG UBUNTU_VERSION=22.04
ARG BUILD_TYPE=base

# ==============================================================================
# Stage 1: Base image with ROS2 Humble
# ==============================================================================
FROM osrf/ros:${ROS_DISTRO}-desktop AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 dependencies
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-v4l2-camera \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: CUDA-enabled image (for GPU support)
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS cuda-base

# Copy ROS2 from base stage
COPY --from=base /opt/ros /opt/ros
COPY --from=base /usr/bin/python3 /usr/bin/python3
COPY --from=base /usr/lib/python3 /usr/lib/python3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-opencv \
    ros-humble-v4l2-camera \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 3: Build stage (installs Python dependencies)
# ==============================================================================
FROM ${BUILD_TYPE} AS builder

ARG BUILD_TYPE

# Set working directory
WORKDIR /tmp/build

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch based on build type
RUN if [ "$BUILD_TYPE" = "cuda-base" ]; then \
        pip3 install torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip3 install torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu \
            --ignore-installed sympy; \
    fi

# Install other Python dependencies
RUN pip3 install --no-cache-dir \
    transformers>=4.35.0 \
    huggingface-hub>=0.19.0 \
    opencv-python>=4.8.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0,<2.0 \
    timm>=0.9.0

# Install Depth Anything 3
RUN pip3 install --no-cache-dir \
    git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# ==============================================================================
# Stage 4: Final runtime image
# ==============================================================================
FROM ${BUILD_TYPE} AS runtime

ARG BUILD_TYPE

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages \
    /usr/local/lib/python3.10/dist-packages

# Create workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Copy package source
COPY . /ros2_ws/src/depth_anything_3_ros2

# Source ROS2 and build package
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --packages-select depth_anything_3_ros2 && \
    rm -rf build log"

# Setup entrypoint
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Environment setup
ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/ros2_ws/install/depth_anything_3_ros2
ENV PYTHONPATH=/ros2_ws/install/depth_anything_3_ros2/lib/python3.10/site-packages:${PYTHONPATH}

# Source ROS2 workspace in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

# Metadata
LABEL maintainer="your@email.com"
LABEL description="Depth Anything 3 ROS2 Wrapper with CUDA support"
LABEL version="1.0.0"
