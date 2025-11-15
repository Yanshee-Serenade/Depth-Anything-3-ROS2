#!/usr/bin/env python3
"""
Benchmark script for Depth Anything 3 ROS2 wrapper.

This script benchmarks different models across various image sizes and
generates performance reports.
"""

import argparse
import time
import json
from pathlib import Path
import sys
from typing import List, Dict
import numpy as np
import cv2
from dataclasses import dataclass, asdict
import platform

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper
except ImportError:
    print("Error: Could not import depth_anything_3_ros2 package")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    device: str
    image_size: tuple
    num_images: int
    avg_inference_time: float
    std_inference_time: float
    min_inference_time: float
    max_inference_time: float
    avg_fps: float
    gpu_memory_allocated: float = None
    gpu_memory_reserved: float = None


class Benchmarker:
    """Benchmark runner for DA3 models."""

    def __init__(self, device: str = 'cuda'):
        """Initialize benchmarker."""
        self.device = device
        self.results: List[BenchmarkResult] = []

    def benchmark_model(
        self,
        model_name: str,
        test_images: List[np.ndarray],
        warmup_runs: int = 3
    ) -> BenchmarkResult:
        """
        Benchmark a single model configuration.

        Args:
            model_name: Model identifier
            test_images: List of test images (numpy arrays)
            warmup_runs: Number of warmup iterations

        Returns:
            BenchmarkResult object
        """
        print(f"\nBenchmarking: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Test images: {len(test_images)}")
        print(f"  Image size: {test_images[0].shape[1]}x{test_images[0].shape[0]}")

        # Load model
        try:
            model = DA3InferenceWrapper(
                model_name=model_name,
                device=self.device
            )
        except Exception as e:
            print(f"  Error loading model: {e}")
            return None

        # Warmup
        print(f"  Warming up ({warmup_runs} iterations)...")
        for i in range(warmup_runs):
            model.inference(test_images[0], return_confidence=False)

        # Benchmark
        print(f"  Running benchmark...")
        inference_times = []

        for img in test_images:
            start = time.time()
            model.inference(img, return_confidence=True)
            inference_time = time.time() - start
            inference_times.append(inference_time)

        # Get GPU memory stats
        gpu_mem = model.get_gpu_memory_usage()

        # Calculate statistics
        inference_times = np.array(inference_times)
        result = BenchmarkResult(
            model_name=model_name,
            device=self.device,
            image_size=(test_images[0].shape[1], test_images[0].shape[0]),
            num_images=len(test_images),
            avg_inference_time=np.mean(inference_times),
            std_inference_time=np.std(inference_times),
            min_inference_time=np.min(inference_times),
            max_inference_time=np.max(inference_times),
            avg_fps=1.0 / np.mean(inference_times),
            gpu_memory_allocated=gpu_mem['allocated_mb'] if gpu_mem else None,
            gpu_memory_reserved=gpu_mem['reserved_mb'] if gpu_mem else None
        )

        # Print results
        print(f"  Results:")
        print(f"    Avg inference: {result.avg_inference_time*1000:.1f} ms")
        print(f"    Std deviation: {result.std_inference_time*1000:.1f} ms")
        print(f"    Min/Max: {result.min_inference_time*1000:.1f} / {result.max_inference_time*1000:.1f} ms")
        print(f"    FPS: {result.avg_fps:.2f}")
        if gpu_mem:
            print(f"    GPU Memory: {result.gpu_memory_allocated:.1f} MB allocated, {result.gpu_memory_reserved:.1f} MB reserved")

        # Cleanup
        del model

        self.results.append(result)
        return result

    def save_results(self, output_file: Path):
        """Save benchmark results to JSON file."""
        data = {
            'system_info': {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            },
            'results': [asdict(r) for r in self.results]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def print_summary(self):
        """Print summary table of all results."""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Model':<30} {'Device':<8} {'Size':<12} {'FPS':<8} {'Time (ms)':<12} {'GPU Mem (MB)':<12}")
        print("-"*80)

        for r in self.results:
            size_str = f"{r.image_size[0]}x{r.image_size[1]}"
            time_str = f"{r.avg_inference_time*1000:.1f}"
            gpu_str = f"{r.gpu_memory_allocated:.0f}" if r.gpu_memory_allocated else "N/A"

            print(f"{r.model_name:<30} {r.device:<8} {size_str:<12} {r.avg_fps:<8.2f} {time_str:<12} {gpu_str:<12}")

        print("="*80)


def load_test_images(
    image_dir: Path,
    max_images: int = 10,
    target_size: tuple = None
) -> List[np.ndarray]:
    """
    Load test images from directory.

    Args:
        image_dir: Directory containing test images
        max_images: Maximum number of images to load
        target_size: Optional (width, height) to resize images

    Returns:
        List of numpy arrays (RGB format)
    """
    images = []
    extensions = ['*.jpg', '*.jpeg', '*.png']

    for ext in extensions:
        for img_path in sorted(image_dir.glob(ext))[:max_images]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if target_size:
                img_rgb = cv2.resize(img_rgb, target_size)

            images.append(img_rgb)

            if len(images) >= max_images:
                break

        if len(images) >= max_images:
            break

    return images


def generate_synthetic_images(
    count: int = 10,
    size: tuple = (640, 480)
) -> List[np.ndarray]:
    """
    Generate synthetic test images.

    Args:
        count: Number of images to generate
        size: Image size as (width, height)

    Returns:
        List of synthetic RGB images
    """
    images = []
    for _ in range(count):
        # Generate random image
        img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        images.append(img)

    return images


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Depth Anything 3 models'
    )
    parser.add_argument(
        '--images',
        type=Path,
        help='Directory containing test images (optional, will use synthetic if not provided)'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='depth-anything/DA3-BASE',
        help='Comma-separated list of models to benchmark'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--sizes',
        type=str,
        default='640x480,1280x720',
        help='Comma-separated list of image sizes (WxH)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=10,
        help='Number of test images to use'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmark_results.json'),
        help='Output file for results'
    )

    args = parser.parse_args()

    # Parse models
    models = [m.strip() for m in args.models.split(',')]

    # Parse sizes
    sizes = []
    for size_str in args.sizes.split(','):
        w, h = map(int, size_str.strip().split('x'))
        sizes.append((w, h))

    print("="*80)
    print("DEPTH ANYTHING 3 BENCHMARK")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Device: {args.device}")
    print(f"Image sizes: {', '.join([f'{w}x{h}' for w, h in sizes])}")
    print(f"Number of test images: {args.num_images}")

    # Initialize benchmarker
    benchmarker = Benchmarker(device=args.device)

    # Run benchmarks
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"Testing with image size: {size[0]}x{size[1]}")
        print(f"{'='*80}")

        # Load or generate test images
        if args.images and args.images.exists():
            print(f"Loading images from: {args.images}")
            test_images = load_test_images(
                args.images,
                max_images=args.num_images,
                target_size=size
            )
        else:
            print(f"Generating {args.num_images} synthetic images")
            test_images = generate_synthetic_images(
                count=args.num_images,
                size=size
            )

        if not test_images:
            print("No test images available, skipping...")
            continue

        # Benchmark each model
        for model_name in models:
            benchmarker.benchmark_model(
                model_name,
                test_images,
                warmup_runs=args.warmup
            )

    # Print summary and save results
    benchmarker.print_summary()
    benchmarker.save_results(args.output)


if __name__ == '__main__':
    main()
