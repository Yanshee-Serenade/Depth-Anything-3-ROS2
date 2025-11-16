#!/bin/bash
# Batch process multiple ROS2 bags through Depth Anything 3
# This script processes ROS2 bag files and saves depth outputs

set -e

# Default values
INPUT_DIR=""
OUTPUT_DIR="depth_output"
MODEL="depth-anything/DA3-BASE"
DEVICE="cuda"
IMAGE_TOPIC="/camera/image_raw"
RECORD_TOPICS="/depth_anything_3/depth /depth_anything_3/depth_colored"

# Function to display usage
usage() {
    echo "Usage: $0 -i INPUT_DIR [-o OUTPUT_DIR] [-m MODEL] [-d DEVICE] [-t TOPIC]"
    echo ""
    echo "Options:"
    echo "  -i INPUT_DIR    Directory containing ROS2 bag files (required)"
    echo "  -o OUTPUT_DIR   Output directory for processed bags (default: depth_output)"
    echo "  -m MODEL        DA3 model to use (default: depth-anything/DA3-BASE)"
    echo "  -d DEVICE       Device: cuda or cpu (default: cuda)"
    echo "  -t TOPIC        Input image topic (default: /camera/image_raw)"
    echo "  -h              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i ./raw_bags -o ./depth_bags -m depth-anything/DA3-LARGE"
    exit 1
}

# Parse arguments
while getopts "i:o:m:d:t:h" opt; do
    case $opt in
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        m) MODEL="$OPTARG" ;;
        d) DEVICE="$OPTARG" ;;
        t) IMAGE_TOPIC="$OPTARG" ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required"
    usage
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ROS2 Batch Processing with Depth Anything 3"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Image topic: $IMAGE_TOPIC"
echo "=========================================="
echo ""

# Find all bag files
BAG_FILES=($(find "$INPUT_DIR" -type d -name "*.db3" -o -name "*.mcap"))

if [ ${#BAG_FILES[@]} -eq 0 ]; then
    echo "Error: No bag files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#BAG_FILES[@]} bag file(s)"
echo ""

# Process each bag
for i in "${!BAG_FILES[@]}"; do
    BAG_PATH="${BAG_FILES[$i]}"
    BAG_NAME=$(basename "$BAG_PATH")
    OUTPUT_BAG="$OUTPUT_DIR/${BAG_NAME}_depth"

    CURRENT=$((i + 1))
    echo "[$CURRENT/${#BAG_FILES[@]}] Processing: $BAG_NAME"
    echo "  Output: $OUTPUT_BAG"

    # Start depth estimation node
    echo "  Starting depth estimation node..."
    ros2 run depth_anything_3_ros2 depth_anything_3_node \
        --ros-args \
        -p model_name:="$MODEL" \
        -p device:="$DEVICE" \
        -r ~/image_raw:="$IMAGE_TOPIC" &
    NODE_PID=$!

    # Wait for node to initialize
    sleep 5

    # Start recording
    echo "  Starting recorder..."
    ros2 bag record $RECORD_TOPICS \
        -o "$OUTPUT_BAG" &
    RECORD_PID=$!

    # Play bag
    echo "  Playing bag file..."
    ros2 bag play "$BAG_PATH" --rate 1.0

    # Stop recording
    echo "  Stopping recorder..."
    kill -SIGINT $RECORD_PID
    wait $RECORD_PID 2>/dev/null || true

    # Stop node
    echo "  Stopping node..."
    kill -SIGTERM $NODE_PID
    wait $NODE_PID 2>/dev/null || true

    echo "  ✓ Completed"
    echo ""
done

echo "=========================================="
echo "Batch processing complete!"
echo "Output directory: $OUTPUT_DIR"
echo "Processed ${#BAG_FILES[@]} bag file(s)"
echo "=========================================="
