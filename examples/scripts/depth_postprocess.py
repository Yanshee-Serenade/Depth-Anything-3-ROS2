#!/usr/bin/env python3
"""
Depth Map Post-Processing Example

Demonstrates various post-processing techniques for depth maps:
- Bilateral filtering for edge-preserving smoothing
- Hole filling for missing depth values
- Temporal filtering for video sequences
- Outlier removal
- Depth inpainting

Requirements:
    pip install opencv-python numpy scipy

Usage:
    python3 depth_postprocess.py --input depth.npy --output processed.npy
    python3 depth_postprocess.py --input depth_dir/ --output output_dir/ --batch
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy import ndimage
import sys


class DepthPostProcessor:
    """Post-processing utilities for depth maps."""

    def __init__(
        self,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        temporal_alpha: float = 0.8,
    ):
        """
        Initialize depth post-processor.

        Args:
            bilateral_d: Diameter of bilateral filter neighborhood
            bilateral_sigma_color: Bilateral filter sigma in color space
            bilateral_sigma_space: Bilateral filter sigma in coordinate space
            temporal_alpha: Temporal smoothing factor (0-1)
        """
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.temporal_alpha = temporal_alpha
        self.prev_depth: Optional[np.ndarray] = None

    def bilateral_filter(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.

        Args:
            depth: Input depth map (H, W) in float32

        Returns:
            Smoothed depth map
        """
        # Normalize to 0-255 for filtering
        depth_normalized = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            depth_normalized,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

        # Convert back to original range
        depth_min, depth_max = depth.min(), depth.max()
        filtered_float = filtered.astype(np.float32) / 255.0
        return filtered_float * (depth_max - depth_min) + depth_min

    def fill_holes(
        self, depth: np.ndarray, max_hole_size: int = 20
    ) -> np.ndarray:
        """
        Fill holes in depth map using morphological operations.

        Args:
            depth: Input depth map with holes (zeros)
            max_hole_size: Maximum hole size to fill in pixels

        Returns:
            Depth map with filled holes
        """
        # Create mask of valid depth values
        mask = (depth > 0).astype(np.uint8)

        # Dilate mask to fill small holes
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max_hole_size, max_hole_size)
        )
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint holes
        depth_normalized = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        inpainted = cv2.inpaint(
            depth_normalized,
            (1 - dilated_mask).astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )

        # Convert back to original range
        depth_min, depth_max = depth[depth > 0].min(), depth.max()
        inpainted_float = inpainted.astype(np.float32) / 255.0
        result = inpainted_float * (depth_max - depth_min) + depth_min

        # Preserve original valid values
        result = np.where(mask, depth, result)
        return result

    def remove_outliers(
        self, depth: np.ndarray, threshold: float = 2.0
    ) -> np.ndarray:
        """
        Remove outliers using statistical filtering.

        Args:
            depth: Input depth map
            threshold: Number of standard deviations for outlier detection

        Returns:
            Depth map with outliers removed
        """
        # Calculate local statistics
        mean_depth = ndimage.uniform_filter(depth, size=5)
        sq_depth = ndimage.uniform_filter(depth**2, size=5)
        std_depth = np.sqrt(np.maximum(sq_depth - mean_depth**2, 0))

        # Identify outliers
        outlier_mask = np.abs(depth - mean_depth) > (threshold * std_depth)

        # Replace outliers with local mean
        result = depth.copy()
        result[outlier_mask] = mean_depth[outlier_mask]
        return result

    def temporal_filter(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing across frames.

        Args:
            depth: Current frame depth map

        Returns:
            Temporally smoothed depth map
        """
        if self.prev_depth is None:
            self.prev_depth = depth.copy()
            return depth

        # Exponential moving average
        smoothed = (
            self.temporal_alpha * self.prev_depth
            + (1 - self.temporal_alpha) * depth
        )
        self.prev_depth = smoothed.copy()
        return smoothed

    def reset_temporal(self):
        """Reset temporal filter state."""
        self.prev_depth = None

    def process_full(
        self,
        depth: np.ndarray,
        enable_bilateral: bool = True,
        enable_hole_fill: bool = True,
        enable_outliers: bool = True,
        enable_temporal: bool = False,
    ) -> np.ndarray:
        """
        Apply full post-processing pipeline.

        Args:
            depth: Input depth map
            enable_bilateral: Enable bilateral filtering
            enable_hole_fill: Enable hole filling
            enable_outliers: Enable outlier removal
            enable_temporal: Enable temporal smoothing

        Returns:
            Fully processed depth map
        """
        result = depth.copy()

        if enable_outliers:
            result = self.remove_outliers(result)

        if enable_hole_fill:
            result = self.fill_holes(result)

        if enable_bilateral:
            result = self.bilateral_filter(result)

        if enable_temporal:
            result = self.temporal_filter(result)

        return result


def load_depth(path: Path) -> np.ndarray:
    """Load depth map from file."""
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix in ['.png', '.jpg', '.jpeg']:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_depth(depth: np.ndarray, path: Path):
    """Save depth map to file."""
    if path.suffix == '.npy':
        np.save(path, depth)
    elif path.suffix in ['.png', '.jpg', '.jpeg']:
        depth_normalized = cv2.normalize(
            depth, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U
        )
        cv2.imwrite(str(path), depth_normalized)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def visualize_comparison(
    original: np.ndarray, processed: np.ndarray, output_path: Path
):
    """Create side-by-side comparison visualization."""
    # Normalize for visualization
    orig_vis = cv2.normalize(
        original, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    proc_vis = cv2.normalize(
        processed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )

    # Apply colormap
    orig_colored = cv2.applyColorMap(orig_vis, cv2.COLORMAP_TURBO)
    proc_colored = cv2.applyColorMap(proc_vis, cv2.COLORMAP_TURBO)

    # Concatenate horizontally
    comparison = np.hstack([orig_colored, proc_colored])

    # Add labels
    cv2.putText(
        comparison, 'Original', (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    cv2.putText(
        comparison, 'Processed',
        (original.shape[1] + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    cv2.imwrite(str(output_path), comparison)
    print(f"Saved comparison: {output_path}")


def process_single(
    input_path: Path,
    output_path: Path,
    processor: DepthPostProcessor,
    args: argparse.Namespace,
):
    """Process single depth map."""
    print(f"Processing: {input_path}")

    # Load depth
    depth = load_depth(input_path)

    # Process
    processed = processor.process_full(
        depth,
        enable_bilateral=args.bilateral,
        enable_hole_fill=args.hole_fill,
        enable_outliers=args.outliers,
        enable_temporal=args.temporal,
    )

    # Save
    save_depth(processed, output_path)
    print(f"Saved: {output_path}")

    # Create visualization if requested
    if args.visualize:
        vis_path = output_path.parent / f"{output_path.stem}_comparison.png"
        visualize_comparison(depth, processed, vis_path)


def process_batch(
    input_dir: Path,
    output_dir: Path,
    processor: DepthPostProcessor,
    args: argparse.Namespace,
):
    """Process directory of depth maps."""
    # Find all depth files
    depth_files = sorted(
        list(input_dir.glob('*.npy'))
        + list(input_dir.glob('*.png'))
        + list(input_dir.glob('*.jpg'))
    )

    if not depth_files:
        print(f"No depth files found in {input_dir}")
        return

    print(f"Found {len(depth_files)} depth files")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    for depth_file in depth_files:
        output_path = output_dir / depth_file.name
        process_single(depth_file, output_path, processor, args)

    print(f"\nProcessed {len(depth_files)} files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Post-process depth maps with various filters'
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Input depth file or directory'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output depth file or directory'
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='Process directory of files'
    )
    parser.add_argument(
        '--bilateral', action='store_true', default=True,
        help='Enable bilateral filtering (default: True)'
    )
    parser.add_argument(
        '--no-bilateral', dest='bilateral', action='store_false',
        help='Disable bilateral filtering'
    )
    parser.add_argument(
        '--hole-fill', action='store_true', default=True,
        help='Enable hole filling (default: True)'
    )
    parser.add_argument(
        '--no-hole-fill', dest='hole_fill', action='store_false',
        help='Disable hole filling'
    )
    parser.add_argument(
        '--outliers', action='store_true', default=True,
        help='Enable outlier removal (default: True)'
    )
    parser.add_argument(
        '--no-outliers', dest='outliers', action='store_false',
        help='Disable outlier removal'
    )
    parser.add_argument(
        '--temporal', action='store_true',
        help='Enable temporal filtering for sequences'
    )
    parser.add_argument(
        '--visualize', '-v', action='store_true',
        help='Create comparison visualizations'
    )
    parser.add_argument(
        '--bilateral-d', type=int, default=9,
        help='Bilateral filter diameter (default: 9)'
    )
    parser.add_argument(
        '--bilateral-sigma-color', type=float, default=75.0,
        help='Bilateral sigma color (default: 75.0)'
    )
    parser.add_argument(
        '--bilateral-sigma-space', type=float, default=75.0,
        help='Bilateral sigma space (default: 75.0)'
    )
    parser.add_argument(
        '--temporal-alpha', type=float, default=0.8,
        help='Temporal smoothing alpha (default: 0.8)'
    )

    args = parser.parse_args()

    # Create processor
    processor = DepthPostProcessor(
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        temporal_alpha=args.temporal_alpha,
    )

    # Process
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.batch:
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)
        process_batch(input_path, output_path, processor, args)
    else:
        if not input_path.is_file():
            print(f"Error: {input_path} does not exist")
            sys.exit(1)
        process_single(input_path, output_path, processor, args)


if __name__ == '__main__':
    main()
