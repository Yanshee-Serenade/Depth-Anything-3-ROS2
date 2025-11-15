#!/bin/bash
# Download sample test images for Depth Anything 3 ROS2 testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="$SCRIPT_DIR/../images"

echo "Creating image directories..."
mkdir -p "$IMAGES_DIR"/{indoor,outdoor,objects}

echo "Downloading sample images..."

# Function to download from URL
download_image() {
    local url=$1
    local output=$2
    echo "  - $(basename $output)"
    wget -q -O "$output" "$url" || curl -s -o "$output" "$url"
}

# Indoor scenes (from Unsplash - free to use)
echo "Downloading indoor scenes..."
download_image \
    "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=1920&q=80" \
    "$IMAGES_DIR/indoor/living_room_01.jpg"

download_image \
    "https://images.unsplash.com/photo-1556912167-f556f1f39faa?w=1920&q=80" \
    "$IMAGES_DIR/indoor/kitchen_01.jpg"

download_image \
    "https://images.unsplash.com/photo-1497366216548-37526070297c?w=1920&q=80" \
    "$IMAGES_DIR/indoor/office_01.jpg"

# Outdoor scenes
echo "Downloading outdoor scenes..."
download_image \
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1920&q=80" \
    "$IMAGES_DIR/outdoor/street_01.jpg"

download_image \
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1920&q=80" \
    "$IMAGES_DIR/outdoor/park_01.jpg"

download_image \
    "https://images.unsplash.com/photo-1486870591958-9b9d0d1dda99?w=1920&q=80" \
    "$IMAGES_DIR/outdoor/building_01.jpg"

# Object scenes
echo "Downloading object scenes..."
download_image \
    "https://images.unsplash.com/photo-1484480974693-6ca0a78fb36b?w=1920&q=80" \
    "$IMAGES_DIR/objects/table_objects.jpg"

download_image \
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1920&q=80" \
    "$IMAGES_DIR/objects/shelf_items.jpg"

download_image \
    "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=1920&q=80" \
    "$IMAGES_DIR/objects/close_up.jpg"

echo ""
echo "Download complete! Sample images saved to:"
echo "  $IMAGES_DIR"
echo ""
echo "Image count:"
find "$IMAGES_DIR" -type f -name "*.jpg" | wc -l | xargs echo "  Total images:"
echo ""
echo "Usage:"
echo "  ros2 launch depth_anything_3_ros2 image_publisher_test.launch.py \\"
echo "    image_path:=$IMAGES_DIR/outdoor/street_01.jpg"
