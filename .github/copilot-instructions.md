# Depth Anything 3 ROS2 Wrapper - AI Coding Instructions

## Project Overview

Camera-agnostic ROS2 (Humble) wrapper for ByteDance's Depth Anything 3 monocular depth estimation. Targets real-time performance (>30 FPS) on NVIDIA Jetson Orin AGX.

## Architecture

```
depth_anything_3_ros2/
├── depth_anything_3_node.py      # Standard ROS2 node (simpler, flexible)
├── depth_anything_3_node_optimized.py  # High-performance node (TensorRT, CUDA streams)
├── da3_inference.py              # HuggingFace model wrapper (DepthAnything3.from_pretrained)
├── da3_inference_optimized.py    # TensorRT/INT8 optimized inference
├── gpu_utils.py                  # CUDA-accelerated upsampling (GPUDepthUpsampler)
└── utils.py                      # Depth normalization, colorization, metrics
```

**Data Flow**: Camera `sensor_msgs/Image` → Node subscribes on `~/image_raw` → `DA3InferenceWrapper.inference()` → Publishes depth on `~/depth`, colored on `~/depth_colored`, confidence on `~/confidence`

## Critical Patterns

### Camera-Agnostic Design (Mandatory)
- **Never** add camera-specific logic to core modules
- Use ROS2 topic remapping for camera integration: `image_topic:=/camera/image_raw`
- Camera configs exist only in `config/camera_configs/*.yaml` and example launch files

### ROS2 Node Structure
```python
# Standard patterns used throughout:
self.declare_parameter('param_name', default_value)  # Always declare with defaults
qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, ...)  # Use BEST_EFFORT for images
self.create_subscription(Image, '~/image_raw', self.callback, qos)  # Relative topics with ~
self.create_publisher(Image, '~/depth', 10)
```

### Inference Wrapper Pattern
```python
# da3_inference.py loads model from HuggingFace:
from depth_anything_3.api import DepthAnything3
self._model = DepthAnything3.from_pretrained(model_name)

# Returns dict: {'depth': np.ndarray, 'confidence': np.ndarray}
result = wrapper.inference(rgb_image, return_confidence=True)
```

## Build & Run Commands

```bash
# Build (source ROS2 first)
source /opt/ros/jazzy/setup.bash  # or humble
colcon build --packages-select depth_anything_3_ros2 --symlink-install
source install/setup.bash

# Run standard node
ros2 launch depth_anything_3_ros2 depth_anything_3.launch.py \
  image_topic:=/camera/image_raw model_name:=depth-anything/DA3-BASE

# Run optimized node (>30 FPS)
ros2 launch depth_anything_3_ros2 depth_anything_3_optimized.launch.py \
  backend:=tensorrt_int8 model_input_height:=384

# Run tests
colcon test --packages-select depth_anything_3_ros2
colcon test-result --verbose
```

## Code Style Requirements

- **PEP 8** with 88 char line length (Black)
- **Google-style docstrings** with type hints on all functions
- **No emojis** in code, docs, or commit messages
- Naming: `PascalCase` classes, `snake_case` functions, `_private_methods`

Example:
```python
def process_image(self, image: np.ndarray, normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    Process input image and return depth estimation.

    Args:
        image: Input RGB image as numpy array (H, W, 3)
        normalize: Whether to normalize depth output

    Returns:
        Dictionary containing depth map and confidence
    """
```

## Key Configuration

- **Models**: `depth-anything/DA3-SMALL`, `DA3-BASE`, `DA3-LARGE`, `DA3-GIANT`
- **Parameters** in `config/params.yaml`: `model_name`, `device` (cuda/cpu), `inference_height/width`, `colormap`
- **Performance**: Use 384x384 input with DA3-SMALL + TensorRT INT8 for >30 FPS

## Testing Patterns

Tests use `unittest` with mocked model loading:
```python
@patch('depth_anything_3_ros2.da3_inference.DepthAnything3')
def test_inference(self, mock_da3):
    mock_model = MagicMock()
    mock_da3.from_pretrained.return_value = mock_model
    # Test logic...
```

## Important Files

- `launch/depth_anything_3.launch.py` - Primary launch with all configurable args
- `launch/multi_camera.launch.py` - Multi-camera namespace isolation pattern
- `OPTIMIZATION_GUIDE.md` - TensorRT conversion and performance tuning
- `docker/README.md` - Docker deployment (CPU/GPU/dev modes)
