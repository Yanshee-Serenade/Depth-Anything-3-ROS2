#!/usr/bin/env python3
"""
Test Depth Anything 3 with static images.

This script allows testing the DA3 model without ROS2, useful for quick
validation and debugging.
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper
    from depth_anything_3_ros2.utils import colorize_depth, normalize_depth
except ImportError:
    print("Error: Could not import depth_anything_3_ros2 package")
    print("Make sure the package is built and sourced:")
    print("  cd ~/ros2_ws")
    print("  colcon build --packages-select depth_anything_3_ros2")
    print("  source install/setup.bash")
    sys.exit(1)


def process_image(
    image_path: Path,
    model: DA3InferenceWrapper,
    output_dir: Path = None,
    colormap: str = 'turbo'
) -> dict:
    """
    Process a single image with DA3.

    Args:
        image_path: Path to input image
        model: DA3 inference wrapper instance
        output_dir: Optional directory to save output images
        colormap: Colormap for visualization

    Returns:
        Dictionary with processing results
    """
    print(f"Processing: {image_path.name}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Error: Could not load image {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    start_time = time.time()
    try:
        result = model.inference(
            image_rgb,
            return_confidence=True,
            return_camera_params=False
        )
    except Exception as e:
        print(f"  Error during inference: {e}")
        return None

    inference_time = time.time() - start_time

    # Get results
    depth_map = result['depth']
    confidence_map = result.get('confidence')

    # Normalize and colorize
    depth_normalized = normalize_depth(depth_map)
    depth_colored = colorize_depth(depth_normalized, colormap=colormap)

    # Print stats
    print(f"  Inference time: {inference_time*1000:.1f} ms")
    print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Save outputs if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        # Save colored depth
        cv2.imwrite(
            str(output_dir / f"{stem}_depth_colored.jpg"),
            depth_colored
        )

        # Save raw depth as numpy
        np.save(
            str(output_dir / f"{stem}_depth_raw.npy"),
            depth_map
        )

        # Save confidence if available
        if confidence_map is not None:
            conf_colored = colorize_depth(confidence_map, colormap='viridis')
            cv2.imwrite(
                str(output_dir / f"{stem}_confidence.jpg"),
                conf_colored
            )

        print(f"  Saved outputs to: {output_dir}")

    return {
        'image_path': image_path,
        'inference_time': inference_time,
        'depth_range': (depth_map.min(), depth_map.max()),
        'image_size': (image.shape[1], image.shape[0])
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test Depth Anything 3 with static images'
    )
    parser.add_argument(
        '--image',
        type=Path,
        help='Single image to process'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help='Directory of images to process'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='depth-anything/DA3-BASE',
        help='Model name or path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--colormap',
        type=str,
        default='turbo',
        help='Colormap for visualization'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.input_dir:
        parser.error('Must specify either --image or --input-dir')

    # Initialize model
    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")

    try:
        model = DA3InferenceWrapper(
            model_name=args.model,
            device=args.device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Collect images to process
    images = []
    if args.image:
        images = [args.image]
    elif args.input_dir:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(args.input_dir.glob(ext))
            images.extend(args.input_dir.glob(ext.upper()))

    if not images:
        print("No images found to process")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process\n")

    # Process images
    results = []
    for img_path in sorted(images):
        result = process_image(
            img_path,
            model,
            output_dir=args.output_dir,
            colormap=args.colormap
        )
        if result:
            results.append(result)
        print()

    # Print summary
    if results:
        avg_time = np.mean([r['inference_time'] for r in results])
        print(f"Summary:")
        print(f"  Processed: {len(results)} images")
        print(f"  Average inference time: {avg_time*1000:.1f} ms")
        print(f"  Average FPS: {1.0/avg_time:.1f}")
        print(f"  Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
