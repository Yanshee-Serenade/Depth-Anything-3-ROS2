# Depth Anything 3 ROS2 Wrapper

## Acknowledgments and Credits

This package would not be possible without the excellent work of the following projects and teams:

### Depth Anything 3
- **Team**: ByteDance Seed Team
- **Repository**: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Paper**: [Depth Anything 3: A New Foundation for Metric and Relative Depth Estimation](https://arxiv.org/abs/2511.10647)
- **Project Page**: https://depth-anything-3.github.io/

This wrapper integrates the state-of-the-art Depth Anything 3 model for monocular depth estimation. All credit for the model architecture and training goes to the original authors.

### Inspiration from Prior ROS2 Wrappers
This package was inspired by the following excellent ROS2 wrapper implementations:

- **Depth Anything V2 ROS2**: [grupo-avispa/depth_anything_v2_ros2](https://github.com/grupo-avispa/depth_anything_v2_ros2)
- **Depth Anything ROS2**: [polatztrk/depth_anything_ros](https://github.com/polatztrk/depth_anything_ros)
- **TensorRT Optimized Wrapper**: [scepter914/DepthAnything-ROS](https://github.com/scepter914/DepthAnything-ROS)

Special thanks to these developers for demonstrating effective patterns for ROS2 integration.

---

## Overview

A production-ready, camera-agnostic ROS2 wrapper for Depth Anything 3 (DA3), providing real-time monocular depth estimation from standard RGB images. This package is designed to work seamlessly with any camera publishing standard `sensor_msgs/Image` messages.

### Key Features

- **Camera-Agnostic Design**: Works with ANY camera publishing standard ROS2 image topics
- **Multiple Model Support**: All DA3 variants (Small, Base, Large, Giant, Nested)
- **CUDA Acceleration**: Optimized for NVIDIA GPUs with automatic CPU fallback
- **Multi-Camera Support**: Run multiple instances for multi-camera setups
- **Real-Time Performance**: Optimized for low latency on Jetson Orin AGX
- **Production Ready**: Comprehensive error handling, logging, and testing
- **Docker Support**: Pre-configured Docker and Docker Compose files
- **Example Images**: Sample test images and benchmark scripts included
- **Performance Profiling**: Built-in benchmarking and profiling tools
- **TensorRT Support**: Optimization scripts for NVIDIA Jetson platforms
- **Post-Processing**: Depth map filtering, hole filling, and enhancement
- **INT8 Quantization**: Model compression for faster inference
- **ONNX Export**: Deploy to various platforms and runtimes
- **Complete Documentation**: Sphinx-based API docs with comprehensive tutorials
- **CI/CD Ready**: GitHub Actions workflow for automated testing and validation
- **Docker Testing**: Automated Docker image validation suite
- **RViz2 Visualization**: Pre-configured visualization setup

### Supported Platforms

- **Primary**: NVIDIA Jetson Orin AGX 64GB (JetPack 6.x)
- **Compatible**: Any system with Ubuntu 22.04, ROS2 Humble, and CUDA 12.x (or CPU)
- **ROS2 Distribution**: Humble Hawksbill
- **Python**: 3.10+

---

## Important: Model Downloads

**This wrapper does NOT require cloning the ByteDance Depth Anything 3 repository.** Models are automatically downloaded from Hugging Face Hub on first run.

### How It Works

- Models download automatically from [Hugging Face Hub](https://huggingface.co/depth-anything) on first use
- Downloaded models are cached in `~/.cache/huggingface/hub/` for reuse
- **Internet connection required** for initial model download
- Subsequent runs use the cached models (no internet needed)

### Offline Operation (Robots Without Internet)

For robots or systems without internet access, pre-download models on a connected machine:

```bash
# On a machine WITH internet connection:
python3 -c "
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
# Download model (only needs to be done once)
AutoImageProcessor.from_pretrained('depth-anything/DA3-BASE')
AutoModelForDepthEstimation.from_pretrained('depth-anything/DA3-BASE')
print('Model downloaded to ~/.cache/huggingface/hub/')
"

# Copy the cache directory to your offline robot:
# On source machine:
tar -czf da3_models.tar.gz -C ~/.cache/huggingface .

# On target robot (via USB drive, SCP, etc.):
mkdir -p ~/.cache/huggingface
tar -xzf da3_models.tar.gz -C ~/.cache/huggingface/
```

Alternatively, set a custom cache directory:

```bash
# Download to specific location
export HF_HOME=/path/to/models
python3 -c "from transformers import AutoModelForDepthEstimation; \
            AutoModelForDepthEstimation.from_pretrained('depth-anything/DA3-BASE')"

# On robot, point to the same location
export HF_HOME=/path/to/models
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py
```

**Available Models:**
- `depth-anything/DA3-SMALL` - Fastest, ~1.5GB download
- `depth-anything/DA3-BASE` - Balanced, ~2.5GB download
- `depth-anything/DA3-LARGE` - Best quality, ~4GB download
- `depth-anything/DA3-GIANT` - Maximum quality, ~6.5GB download

---

## Table of Contents

- [Important: Model Downloads](#important-model-downloads)
  - [How It Works](#how-it-works)
  - [Offline Operation](#offline-operation-robots-without-internet)
- [Installation](#installation)
  - [Native Installation](#installation)
  - [Docker Installation](#docker-deployment)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Docker Deployment](#docker-deployment)
- [Example Images and Benchmarks](#example-images-and-benchmarks)
- [Performance](#performance)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

1. **ROS2 Humble** on Ubuntu 22.04:
```bash
# If not already installed
sudo apt update
sudo apt install ros-humble-desktop
```

2. **CUDA 12.x** (optional, for GPU acceleration):
```bash
# For Jetson Orin AGX, this comes with JetPack 6.x
# For desktop systems, install CUDA Toolkit from NVIDIA
nvidia-smi  # Verify CUDA installation
```

3. **Internet Connection** (for initial model download):
- Required during first run to download models from Hugging Face Hub
- See [Offline Operation](#offline-operation-robots-without-internet) if deploying to robots without internet

### Step 1: Install ROS2 Dependencies

```bash
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs \
  ros-humble-image-transport \
  ros-humble-rclpy
```

### Step 2: Install Python Dependencies

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv ~/da3_venv
source ~/da3_venv/bin/activate

# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip3 install transformers>=4.35.0 \
  huggingface-hub>=0.19.0 \
  opencv-python>=4.8.0 \
  pillow>=10.0.0 \
  numpy>=1.24.0 \
  timm>=0.9.0

# Install Depth Anything 3 from source
pip3 install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

**Note**: For CPU-only systems, install PyTorch without CUDA:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Clone and Build Package

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws/src  # Or create: mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# Clone this repository
git clone https://github.com/yourusername/depth_anything_3_ros2.git

# Build the package
cd ~/ros2_ws
colcon build --packages-select depth_anything_3_ros2

# Source the workspace
source install/setup.bash
```

### Step 4: Verify Installation

```bash
# Test that the package is found
ros2 pkg list | grep depth_anything_3_ros2

# Run tests (optional)
colcon test --packages-select depth_anything_3_ros2
colcon test-result --verbose
```

### Step 5: Pre-Download Models (Optional but Recommended)

Pre-download models to avoid delays on first run. **This step is REQUIRED if deploying to offline robots.**

```bash
# Download a specific model (requires internet connection)
python3 -c "
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
print('Downloading DA3-BASE model...')
AutoImageProcessor.from_pretrained('depth-anything/DA3-BASE')
AutoModelForDepthEstimation.from_pretrained('depth-anything/DA3-BASE')
print('Model cached to ~/.cache/huggingface/hub/')
print('You can now run offline!')
"

# For offline robots, copy the cache:
# tar -czf da3_models.tar.gz -C ~/.cache/huggingface .
# Transfer da3_models.tar.gz to robot and extract:
# tar -xzf da3_models.tar.gz -C ~/.cache/huggingface/
```

**Alternative models:**
- For faster inference: Replace `DA3-BASE` with `DA3-SMALL`
- For best quality: Replace `DA3-BASE` with `DA3-LARGE`

See [Model Downloads](#important-model-downloads) for complete offline deployment instructions.

---

## Quick Start

### Single Camera (Generic USB Camera)

The fastest way to get started is with a standard USB camera:

```bash
# Terminal 1: Launch USB camera driver
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p image_size:="[640,480]" \
  -r __ns:=/camera

# Terminal 2: Launch Depth Anything 3
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  image_topic:=/camera/image_raw \
  model_name:=depth-anything/DA3-BASE \
  device:=cuda

# Terminal 3: Visualize with RViz2
rviz2 -d $(ros2 pkg prefix depth_anything_3_ros2)/share/depth_anything_3_ros2/rviz/depth_view.rviz
```

### Using Pre-Built Example Launch Files

```bash
# USB camera example (requires v4l2_camera)
ros2 launch depth_anything_3_ros2 usb_camera_example.launch.py

# Static image test (requires image_publisher)
ros2 launch depth_anything_3_ros2 image_publisher_test.launch.py \
  image_path:=/path/to/your/test_image.jpg
```

---

## Configuration

### Parameters

All parameters can be configured via launch files or command line:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | string | `depth-anything/DA3-BASE` | Hugging Face model ID or local path |
| `device` | string | `cuda` | Inference device (`cuda` or `cpu`) |
| `cache_dir` | string | `""` | Model cache directory (empty for default) |
| `inference_height` | int | `518` | Height for inference (model input) |
| `inference_width` | int | `518` | Width for inference (model input) |
| `input_encoding` | string | `bgr8` | Expected input encoding (`bgr8` or `rgb8`) |
| `normalize_depth` | bool | `true` | Normalize depth to [0, 1] range |
| `publish_colored` | bool | `true` | Publish colorized depth visualization |
| `publish_confidence` | bool | `true` | Publish confidence map |
| `colormap` | string | `turbo` | Colormap for visualization |
| `queue_size` | int | `1` | Subscriber queue size |
| `log_inference_time` | bool | `false` | Log performance metrics |

### Available Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| `depth-anything/DA3-SMALL` | 0.08B | Fast inference, lower accuracy |
| `depth-anything/DA3-BASE` | 0.12B | Balanced performance (recommended) |
| `depth-anything/DA3-LARGE` | 0.35B | Higher accuracy |
| `depth-anything/DA3-GIANT` | 1.15B | Best accuracy, slower |
| `depth-anything/DA3NESTED-GIANT-LARGE` | Combined | Metric scale reconstruction |

### Topics

#### Subscribed Topics
- `~/image_raw` (sensor_msgs/Image): Input RGB image from camera
- `~/camera_info` (sensor_msgs/CameraInfo): Optional camera intrinsics

#### Published Topics
- `~/depth` (sensor_msgs/Image): Depth map (32FC1 encoding)
- `~/depth_colored` (sensor_msgs/Image): Colorized depth visualization (BGR8)
- `~/confidence` (sensor_msgs/Image): Confidence map (32FC1)
- `~/depth/camera_info` (sensor_msgs/CameraInfo): Camera info for depth image

---

## Usage Examples

### Example 1: Generic USB Camera (v4l2_camera)

Complete example with a standard USB webcam:

```bash
# Install v4l2_camera if not already installed
sudo apt install ros-humble-v4l2-camera

# Launch everything together
ros2 launch depth_anything_3_ros2 usb_camera_example.launch.py \
  video_device:=/dev/video0 \
  model_name:=depth-anything/DA3-BASE
```

### Example 2: ZED Stereo Camera

Connect to a ZED camera (requires separate ZED ROS2 wrapper installation):

```bash
# Launch ZED camera separately
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedxm

# In another terminal, launch depth estimation with topic remapping
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  image_topic:=/zed/zed_node/rgb/image_rect_color \
  camera_info_topic:=/zed/zed_node/rgb/camera_info
```

Or use the provided example:
```bash
ros2 launch depth_anything_3_ros2 zed_camera_example.launch.py \
  camera_model:=zedxm
```

### Example 3: Intel RealSense Camera

Connect to a RealSense camera (requires realsense-ros):

```bash
# Launch RealSense camera
ros2 launch realsense2_camera rs_launch.py

# Launch depth estimation
ros2 launch depth_anything_3_ros2 realsense_example.launch.py
```

### Example 4: Multi-Camera Setup

Run depth estimation on 4 cameras simultaneously:

```bash
# Launch multi-camera setup
ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
  camera_namespaces:="cam1,cam2,cam3,cam4" \
  image_topics:="/cam1/image_raw,/cam2/image_raw,/cam3/image_raw,/cam4/image_raw" \
  model_name:=depth-anything/DA3-BASE
```

### Example 5: Testing with Static Images

Test with a static image using image_publisher:

```bash
sudo apt install ros-humble-image-publisher

ros2 launch depth_anything_3_ros2 image_publisher_test.launch.py \
  image_path:=/path/to/test_image.jpg \
  model_name:=depth-anything/DA3-BASE
```

### Example 6: Using Different Models

Switch between models for different performance/accuracy tradeoffs:

```bash
# Fast inference (DA3-Small)
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  model_name:=depth-anything/DA3-SMALL \
  image_topic:=/camera/image_raw

# Best accuracy (DA3-Giant) - requires more GPU memory
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  model_name:=depth-anything/DA3-GIANT \
  image_topic:=/camera/image_raw
```

### Example 7: CPU-Only Mode

Run on systems without CUDA:

```bash
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  image_topic:=/camera/image_raw \
  model_name:=depth-anything/DA3-BASE \
  device:=cpu
```

### Example 8: Custom Configuration

Use a custom parameter file:

```bash
# Create custom config file
cat > my_config.yaml <<EOF
depth_anything_3:
  ros__parameters:
    model_name: "depth-anything/DA3-LARGE"
    device: "cuda"
    normalize_depth: true
    publish_colored: true
    colormap: "viridis"
    log_inference_time: true
EOF

# Launch with custom config
ros2 run depth_anything_3_ros2 depth_anything_3_node --ros-args \
  --params-file my_config.yaml \
  -r ~/image_raw:=/camera/image_raw
```

---

## Docker Deployment

Docker images are provided for easy deployment on both CPU and GPU systems.

### Quick Start with Docker Compose

```bash
# CPU-only mode
docker-compose up -d depth-anything-3-cpu
docker exec -it da3_ros2_cpu bash

# GPU mode (requires nvidia-docker)
docker-compose up -d depth-anything-3-gpu
docker exec -it da3_ros2_gpu bash

# Development mode (source mounted)
docker-compose up -d depth-anything-3-dev
```

### Manual Docker Build

```bash
# Build GPU image
docker build -t depth_anything_3_ros2:gpu \
    --build-arg BUILD_TYPE=cuda-base \
    .

# Run with USB camera
docker run -it --rm \
    --runtime=nvidia \
    --gpus all \
    --network host \
    --privileged \
    -v /dev:/dev:rw \
    depth_anything_3_ros2:gpu
```

### Pre-configured Services

The docker-compose.yml includes:
- `depth-anything-3-cpu`: CPU-only deployment
- `depth-anything-3-gpu`: GPU-accelerated deployment
- `depth-anything-3-dev`: Development environment
- `depth-anything-3-usb-camera`: Standalone USB camera service

### Docker Testing and Validation

Automated test suite for validating Docker images:

```bash
cd docker
chmod +x test_docker.sh
./test_docker.sh
```

This comprehensive test suite validates:
- Docker and Docker Compose installation
- CPU and GPU image builds
- ROS2 installation and package builds
- Python dependencies
- CUDA availability (GPU images)
- Volume mounts and networking
- Model download capability

For detailed Docker documentation, see [docker/README.md](docker/README.md).

---

## Example Images and Benchmarks

### Sample Test Images

Download sample images for quick testing:

```bash
cd examples
./scripts/download_samples.sh
```

This downloads sample indoor, outdoor, and object images from public datasets.

### Testing with Static Images

```bash
# Test single image
python3 examples/scripts/test_with_images.py \
    --image examples/images/outdoor/street_01.jpg \
    --model depth-anything/DA3-BASE \
    --device cuda \
    --output-dir results/

# Batch process directory
python3 examples/scripts/test_with_images.py \
    --input-dir examples/images/outdoor/ \
    --output-dir results/ \
    --model depth-anything/DA3-BASE
```

### Performance Benchmarking

Run comprehensive benchmarks across multiple models and image sizes:

```bash
# Benchmark multiple models
python3 examples/scripts/benchmark.py \
    --images examples/images/ \
    --models depth-anything/DA3-SMALL,depth-anything/DA3-BASE,depth-anything/DA3-LARGE \
    --sizes 640x480,1280x720 \
    --device cuda \
    --output benchmark_results.json
```

Example output:
```
================================================================================
BENCHMARK SUMMARY
================================================================================
Model                          Device   Size         FPS      Time (ms)    GPU Mem (MB)
--------------------------------------------------------------------------------
depth-anything/DA3-SMALL       cuda     640x480      25.3     39.5         1512
depth-anything/DA3-BASE        cuda     640x480      19.8     50.5         2489
depth-anything/DA3-LARGE       cuda     640x480      11.7     85.4         3952
================================================================================
```

### Advanced Example Scripts

#### Depth Post-Processing

Apply filtering, hole filling, and enhancement to depth maps:

```bash
cd examples/scripts

# Process single depth map
python3 depth_postprocess.py \
    --input depth.npy \
    --output processed.npy \
    --visualize

# Batch process directory
python3 depth_postprocess.py \
    --input depth_dir/ \
    --output processed_dir/ \
    --batch
```

#### Multi-Camera Synchronization

Synchronize depth estimation from multiple cameras:

```bash
# Terminal 1: Launch multi-camera setup
ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
    camera_namespaces:=cam_left,cam_right \
    image_topics:=/cam_left/image_raw,/cam_right/image_raw

# Terminal 2: Run synchronizer
python3 multi_camera_sync.py \
    --cameras cam_left cam_right \
    --sync-threshold 0.05 \
    --output synchronized_depth/
```

#### TensorRT Optimization (Jetson)

Optimize models for maximum performance on Jetson platforms:

```bash
# Optimize model
python3 optimize_tensorrt.py \
    --model depth-anything/DA3-BASE \
    --output da3_base_trt.pth \
    --precision fp16 \
    --benchmark

# Expected speedup: 2-3x faster inference
```

#### Performance Tuning

Quantization, ONNX export, and profiling:

```bash
# INT8 quantization
python3 performance_tuning.py quantize \
    --model depth-anything/DA3-BASE \
    --output da3_base_int8.pth

# Export to ONNX
python3 performance_tuning.py export-onnx \
    --model depth-anything/DA3-BASE \
    --output da3_base.onnx \
    --benchmark

# Profile layers
python3 performance_tuning.py profile \
    --model depth-anything/DA3-BASE \
    --layers \
    --memory
```

#### ROS2 Batch Processing

Process ROS2 bags through depth estimation:

```bash
./ros2_batch_process.sh \
    -i ./raw_bags \
    -o ./depth_bags \
    -m depth-anything/DA3-BASE \
    -d cuda
```

#### Node Profiling

Profile ROS2 node performance:

```bash
python3 profile_node.py \
    --model depth-anything/DA3-BASE \
    --device cuda \
    --duration 60
```

For more examples, see [examples/README.md](examples/README.md).

---

## Documentation

Complete documentation is available in multiple formats:

### Sphinx Documentation

Build and view the complete API documentation:

```bash
cd docs
pip install -r requirements.txt
make html
open build/html/index.html  # or xdg-open on Linux
```

### Documentation Contents

- **API Reference**: Complete API documentation with examples
  - [DA3 Inference Module](docs/source/api/da3_inference.rst)
  - [ROS2 Node Module](docs/source/api/depth_anything_3_node.rst)
  - [Utilities Module](docs/source/api/utils.rst)

- **User Guides**:
  - Installation and setup
  - Camera integration guide
  - Multi-camera configuration
  - Performance optimization
  - Troubleshooting

- **Tutorials**:
  - [Quick Start Tutorial](docs/source/tutorials/quick_start.rst) - Get up and running in minutes
  - [USB Camera Setup](docs/source/tutorials/usb_camera.rst) - Complete USB camera guide
  - [Multi-Camera Setup](docs/source/tutorials/multi_camera.rst) - Synchronized multi-camera depth
  - [Performance Tuning](docs/source/tutorials/performance_tuning.rst) - Optimization guide for all platforms

### Additional Documentation

- [Docker Deployment Guide](docker/README.md)
- [Example Images Guide](examples/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Validation Checklist](VALIDATION_CHECKLIST.md)

---

## Performance

### Benchmark Results (Jetson Orin AGX 64GB)

Tested with 640x480 input images:

| Model | FPS | Inference Time | GPU Memory |
|-------|-----|----------------|------------|
| DA3-Small | ~25 FPS | ~40ms | ~1.5 GB |
| DA3-Base | ~20 FPS | ~50ms | ~2.5 GB |
| DA3-Large | ~12 FPS | ~85ms | ~4.0 GB |
| DA3-Giant | ~6 FPS | ~165ms | ~6.5 GB |

### Optimization Tips

1. **TensorRT Optimization** (Jetson platforms):
```bash
cd examples/scripts
python3 optimize_tensorrt.py --model depth-anything/DA3-BASE \
    --output da3_base_trt.pth --precision fp16
# Expected: 2-3x speedup
```

2. **INT8 Quantization** for faster inference:
```bash
python3 performance_tuning.py quantize \
    --model depth-anything/DA3-BASE --output da3_base_int8.pth
# 50-75% smaller, 20-40% faster
```

3. **Reduce Input Resolution**: Lower resolution images process faster
```bash
--param inference_height:=384 inference_width:=512
```

4. **Use Smaller Models**: DA3-SMALL offers best speed, DA3-BASE balances speed/accuracy

5. **Queue Size**: Set to 1 to always process latest frame
```bash
--param queue_size:=1
```

6. **Disable Unused Outputs**: Save processing time
```bash
--param publish_colored_depth:=false
--param publish_confidence:=false
```

7. **Multiple Cameras**: Each camera runs in separate process with shared GPU

8. **Performance Profiling**: Profile to identify bottlenecks
```bash
python3 examples/scripts/profile_node.py --model depth-anything/DA3-BASE
```

For comprehensive optimization guide, see [Performance Tuning Tutorial](docs/source/tutorials/performance_tuning.rst).

---

## Troubleshooting

### Common Issues

#### 1. Model Download Failures

**Error**: `Failed to load model from Hugging Face Hub` or `Connection timeout`

**Solutions**:
- **Check internet connection**: `ping huggingface.co`
- **Verify Hugging Face Hub is accessible**: May be blocked by firewall/proxy
- **Pre-download models manually**:
  ```bash
  python3 -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation; \
              AutoImageProcessor.from_pretrained('depth-anything/DA3-BASE'); \
              AutoModelForDepthEstimation.from_pretrained('depth-anything/DA3-BASE')"
  ```
- **Use custom cache directory**: Set `HF_HOME=/path/to/models` environment variable
- **For offline robots**: See [Offline Operation](#offline-operation-robots-without-internet) section

#### 2. Model Not Found on Offline Robot

**Error**: `Model depth-anything/DA3-BASE not found` on robot without internet

**Solution**: Pre-download models and copy cache directory:
```bash
# On development machine WITH internet:
python3 -c "from transformers import AutoModelForDepthEstimation; \
            AutoModelForDepthEstimation.from_pretrained('depth-anything/DA3-BASE')"
tar -czf da3_models.tar.gz -C ~/.cache/huggingface .

# Transfer to robot (USB, SCP, etc.) and extract:
ssh robot@robot-ip
mkdir -p ~/.cache/huggingface
tar -xzf da3_models.tar.gz -C ~/.cache/huggingface/
```

Verify models are available:
```bash
ls ~/.cache/huggingface/hub/models--depth-anything--*
```

#### 3. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Use a smaller model (DA3-Small or DA3-Base)
- Reduce input resolution
- Close other GPU applications
- Switch to CPU mode temporarily

```bash
# Use smaller model
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  model_name:=depth-anything/DA3-SMALL

# Or use CPU
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  device:=cpu
```

#### 2. Model Download Failures

**Error**: `Failed to load model from Hugging Face Hub`

**Solutions**:
- Check internet connection
- Verify Hugging Face Hub is accessible
- Download model manually and use local path

```bash
# Download manually
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3-BASE')"

# Use local path
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  model_name:=/path/to/local/model
```

#### 3. Image Encoding Mismatches

**Error**: `CV Bridge conversion failed`

**Solutions**:
- Check camera's output encoding
- Adjust `input_encoding` parameter

```bash
# For RGB cameras
--param input_encoding:=rgb8

# For BGR cameras (most common)
--param input_encoding:=bgr8
```

#### 4. No Image Received

**Solutions**:
- Verify camera is publishing: `ros2 topic echo /camera/image_raw`
- Check topic remapping is correct
- Verify QoS settings match camera

```bash
# List available topics
ros2 topic list | grep image

# Check topic info
ros2 topic info /camera/image_raw
```

#### 5. Low Frame Rate

**Solutions**:
- Check GPU utilization: `nvidia-smi`
- Enable performance logging
- Reduce image resolution
- Use smaller model

```bash
# Enable performance logging
--param log_inference_time:=true
```

---

## Development

### Running Tests

```bash
# Run all tests
cd ~/ros2_ws
colcon test --packages-select depth_anything_3_ros2

# View test results
colcon test-result --verbose

# Run specific test
python3 -m pytest src/depth_anything_3_ros2/test/test_inference.py -v
```

### Code Style

This package follows:
- PEP 8 for Python code
- Google-style docstrings
- Type hints for all functions
- No emojis in code or documentation

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Follow code style guidelines
4. Add tests for new functionality
5. Submit a pull request

---

## Citation

If you use Depth Anything 3 in your research, please cite the original paper:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: A New Foundation for Metric and Relative Depth Estimation},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2024}
}
```

---

## License

This ROS2 wrapper is released under the MIT License.

The Depth Anything 3 model has its own license. Please refer to the [official repository](https://github.com/ByteDance-Seed/Depth-Anything-3) for model license information.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/depth_anything_3_ros2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/depth_anything_3_ros2/discussions)
- **ROS2 Documentation**: [ROS2 Humble Docs](https://docs.ros.org/en/humble/)
- **Depth Anything 3**: [Official Repository](https://github.com/ByteDance-Seed/Depth-Anything-3)

---

**Note**: This is an unofficial ROS2 wrapper. For the official Depth Anything 3 implementation, please visit the [ByteDance-Seed repository](https://github.com/ByteDance-Seed/Depth-Anything-3).
