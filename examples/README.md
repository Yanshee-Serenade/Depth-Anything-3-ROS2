# Example Images and Test Data

This directory contains sample images and scripts for testing the Depth Anything 3 ROS2 wrapper.

## Directory Structure

```
examples/
├── images/              # Sample test images
│   ├── indoor/         # Indoor scenes
│   ├── outdoor/        # Outdoor scenes
│   └── objects/        # Object-focused images
├── scripts/            # Utility scripts
│   ├── download_samples.sh
│   ├── test_with_images.py
│   └── benchmark.py
└── README.md
```

## Quick Start

### Download Sample Images

```bash
cd examples
./scripts/download_samples.sh
```

This will download sample images from public datasets:
- KITTI (driving scenes)
- NYU Depth V2 (indoor scenes)
- Sample images from Depth Anything paper

### Test with Static Images

```bash
# Single image test
python3 scripts/test_with_images.py \
    --image images/outdoor/street_01.jpg \
    --model depth-anything/DA3-BASE \
    --device cuda

# Batch processing
python3 scripts/test_with_images.py \
    --input-dir images/outdoor/ \
    --output-dir results/ \
    --model depth-anything/DA3-BASE
```

### Benchmark Performance

```bash
# Run benchmark on all test images
python3 scripts/benchmark.py \
    --images images/ \
    --models DA3-SMALL,DA3-BASE,DA3-LARGE \
    --device cuda \
    --output benchmark_results.json
```

## Sample Images Included

### Indoor Scenes
- `living_room_01.jpg` - Living room with furniture
- `kitchen_01.jpg` - Kitchen scene
- `office_01.jpg` - Office workspace

### Outdoor Scenes
- `street_01.jpg` - Urban street scene
- `park_01.jpg` - Park landscape
- `building_01.jpg` - Building facade

### Object Scenes
- `table_objects.jpg` - Objects on table
- `shelf_items.jpg` - Items on shelf
- `close_up.jpg` - Close-up object

## Using with ROS2

### Method 1: image_publisher

```bash
# Publish static image to ROS2
ros2 run image_publisher image_publisher_node \
    examples/images/outdoor/street_01.jpg \
    --ros-args -p publish_rate:=1.0

# In another terminal, run depth estimation
ros2 run depth_anything_3_ros2 depth_anything_3_node \
    --ros-args -r ~/image_raw:=/image_raw
```

### Method 2: Launch File

```bash
ros2 launch depth_anything_3_ros2 image_publisher_test.launch.py \
    image_path:=$(pwd)/examples/images/outdoor/street_01.jpg \
    model_name:=depth-anything/DA3-BASE
```

### Method 3: Batch Processing Script

```bash
# Process all images in a directory
./scripts/ros2_batch_process.sh images/outdoor/ results/
```

## Image Attribution

Sample images are sourced from:

1. **Public Domain Images**
   - Unsplash (https://unsplash.com)
   - Pexels (https://pexels.com)

2. **Research Datasets**
   - KITTI Vision Benchmark Suite (http://www.cvlibs.net/datasets/kitti/)
   - NYU Depth Dataset V2 (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

3. **Depth Anything Examples**
   - From official Depth Anything 3 repository

All images are used for testing and demonstration purposes only.

## Adding Your Own Images

```bash
# Place your images in appropriate directory
cp my_image.jpg examples/images/custom/

# Test with your image
ros2 launch depth_anything_3_ros2 image_publisher_test.launch.py \
    image_path:=examples/images/custom/my_image.jpg
```

## Best Practices for Test Images

1. **Resolution**: 640x480 to 1920x1080 recommended
2. **Format**: JPG, PNG supported
3. **Content**: Varied scenes for comprehensive testing
4. **Quality**: Good lighting, minimal motion blur

## Troubleshooting

### Image Not Found
```bash
# Check image exists
ls -lh examples/images/outdoor/street_01.jpg

# Use absolute path
ros2 launch depth_anything_3_ros2 image_publisher_test.launch.py \
    image_path:=/full/path/to/image.jpg
```

### Unsupported Format
```bash
# Convert to supported format
convert input.bmp output.jpg
```

## Creating Custom Test Sets

```bash
# Create directory for your test set
mkdir -p examples/images/my_test_set

# Add images
cp /path/to/images/*.jpg examples/images/my_test_set/

# Run benchmark
python3 scripts/benchmark.py \
    --images examples/images/my_test_set/ \
    --output my_results.json
```

## License

Sample images maintain their original licenses. Please check individual image metadata or LICENSE files in subdirectories for specific attribution requirements.
