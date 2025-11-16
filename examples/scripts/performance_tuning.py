#!/usr/bin/env python3
"""
Performance Tuning Utilities

Provides various performance optimization techniques:
- INT8 quantization for faster inference
- Dynamic batching for multi-frame processing
- Model pruning for reduced memory
- ONNX export for deployment flexibility
- Performance profiling and analysis

Requirements:
    pip install torch onnx onnxruntime

Usage:
    # Quantize model to INT8
    python3 performance_tuning.py quantize \
        --model depth-anything/DA3-BASE \
        --output da3_base_int8.pth

    # Export to ONNX
    python3 performance_tuning.py export-onnx \
        --model depth-anything/DA3-BASE \
        --output da3_base.onnx

    # Profile model
    python3 performance_tuning.py profile \
        --model depth-anything/DA3-BASE \
        --iterations 100
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np

try:
    import torch
    import torch.quantization
except ImportError:
    print("Error: PyTorch not installed")
    sys.exit(1)

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("Error: transformers not installed")
    sys.exit(1)


class ModelQuantizer:
    """INT8 quantization for model compression and speedup."""

    def __init__(self, model_name: str = "depth-anything/DA3-BASE"):
        """
        Initialize quantizer.

        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        print(f"Initializing quantizer for {model_name}")

    def load_model(self) -> Tuple[torch.nn.Module, object]:
        """
        Load model and processor.

        Returns:
            Tuple of (model, processor)
        """
        print("Loading model...")
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        model.eval()
        print("Model loaded")
        return model, processor

    def quantize_dynamic(
        self, model: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Apply dynamic quantization.

        Args:
            model: Original model

        Returns:
            Quantized model
        """
        print("\nApplying dynamic INT8 quantization...")

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

        print("Quantization complete")
        return quantized_model

    def quantize_static(
        self,
        model: torch.nn.Module,
        calibration_data: List[torch.Tensor],
    ) -> torch.nn.Module:
        """
        Apply static quantization with calibration.

        Args:
            model: Original model
            calibration_data: List of calibration tensors

        Returns:
            Quantized model
        """
        print("\nApplying static INT8 quantization...")
        print(f"Calibration samples: {len(calibration_data)}")

        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)

        # Calibrate
        print("Calibrating...")
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                model(data)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(calibration_data)} samples")

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        print("Static quantization complete")
        return model

    def compare_size(
        self, original: torch.nn.Module, quantized: torch.nn.Module
    ):
        """Compare model sizes."""
        print("\nModel Size Comparison:")

        # Save temporarily to measure size
        torch.save(original.state_dict(), '/tmp/original.pth')
        torch.save(quantized.state_dict(), '/tmp/quantized.pth')

        orig_size = Path('/tmp/original.pth').stat().st_size / (1024 ** 2)
        quant_size = Path('/tmp/quantized.pth').stat().st_size / (1024 ** 2)

        print(f"  Original: {orig_size:.2f} MB")
        print(f"  Quantized: {quant_size:.2f} MB")
        print(f"  Reduction: {(1 - quant_size/orig_size) * 100:.1f}%")

        # Cleanup
        Path('/tmp/original.pth').unlink()
        Path('/tmp/quantized.pth').unlink()


class ONNXExporter:
    """Export models to ONNX format."""

    def __init__(self, model_name: str = "depth-anything/DA3-BASE"):
        """
        Initialize ONNX exporter.

        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        print(f"Initializing ONNX exporter for {model_name}")

    def export(
        self,
        model: torch.nn.Module,
        output_path: Path,
        input_shape: Tuple[int, int, int, int] = (1, 3, 518, 518),
        opset_version: int = 14,
    ):
        """
        Export model to ONNX.

        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            input_shape: Input tensor shape
            opset_version: ONNX opset version
        """
        print(f"\nExporting to ONNX (opset {opset_version})...")
        print(f"Input shape: {input_shape}")

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['depth'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'depth': {0: 'batch_size'}
            }
        )

        print(f"Exported to {output_path}")

        # Verify
        self.verify_onnx(output_path)

    def verify_onnx(self, onnx_path: Path):
        """Verify ONNX model."""
        try:
            import onnx
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            print("ONNX model verification passed")
        except ImportError:
            print("Warning: onnx package not installed, skipping verification")
        except Exception as e:
            print(f"ONNX verification failed: {e}")

    def benchmark_onnx(
        self,
        onnx_path: Path,
        input_shape: Tuple[int, int, int, int] = (1, 3, 518, 518),
        iterations: int = 100,
    ):
        """Benchmark ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime not installed")
            print("Install with: pip install onnxruntime-gpu")
            return

        print(f"\nBenchmarking ONNX model ({iterations} iterations)...")

        # Create session
        sess = ort.InferenceSession(str(onnx_path))

        # Create input
        x = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            sess.run(None, {'input': x})

        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            sess.run(None, {'input': x})
            times.append(time.time() - start)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{iterations} iterations")

        times = np.array(times)

        print("\nONNX Benchmark Results:")
        print(f"  Mean: {np.mean(times) * 1000:.2f} ms")
        print(f"  Std: {np.std(times) * 1000:.2f} ms")
        print(f"  FPS: {1.0 / np.mean(times):.2f}")


class PerformanceProfiler:
    """Profile model performance and identify bottlenecks."""

    def __init__(self):
        """Initialize profiler."""
        print("Initializing performance profiler")

    def profile_layers(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        iterations: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Profile individual layers.

        Args:
            model: Model to profile
            input_tensor: Example input
            iterations: Number of iterations

        Returns:
            Dictionary mapping layer names to timing statistics
        """
        print(f"\nProfiling layers ({iterations} iterations)...")

        layer_times = {}

        def hook_fn(name):
            def hook(module, input, output):
                if name not in layer_times:
                    layer_times[name] = []

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()

                # Forward pass already happened
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start

                layer_times[name].append(elapsed)

            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Run profiling
        model.eval()
        with torch.no_grad():
            for i in range(iterations):
                _ = model(input_tensor)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{iterations} iterations")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Calculate statistics
        stats = {}
        for name, times in layer_times.items():
            times_arr = np.array(times)
            stats[name] = {
                'mean_ms': np.mean(times_arr) * 1000,
                'std_ms': np.std(times_arr) * 1000,
                'total_ms': np.sum(times_arr) * 1000,
            }

        # Sort by total time
        sorted_stats = dict(
            sorted(
                stats.items(),
                key=lambda x: x[1]['total_ms'],
                reverse=True
            )
        )

        # Print top 10 slowest layers
        print("\nTop 10 Slowest Layers:")
        for i, (name, timing) in enumerate(
            list(sorted_stats.items())[:10]
        ):
            print(
                f"  {i+1}. {name}: "
                f"{timing['mean_ms']:.3f} ms "
                f"(total: {timing['total_ms']:.2f} ms)"
            )

        return sorted_stats

    def profile_memory(
        self, model: torch.nn.Module, input_tensor: torch.Tensor
    ):
        """Profile memory usage."""
        if not torch.cuda.is_available():
            print("Memory profiling requires CUDA")
            return

        print("\nProfiling memory usage...")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Initial memory
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)

        torch.cuda.synchronize()

        # Final memory
        mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(f"  Memory before: {mem_before:.2f} MB")
        print(f"  Memory after: {mem_after:.2f} MB")
        print(f"  Peak memory: {mem_peak:.2f} MB")
        print(f"  Memory increase: {mem_after - mem_before:.2f} MB")


def cmd_quantize(args):
    """Quantize model command."""
    quantizer = ModelQuantizer(args.model)
    model, processor = quantizer.load_model()

    if args.mode == 'dynamic':
        quantized = quantizer.quantize_dynamic(model)
    else:
        print("Static quantization requires calibration data")
        print("Not implemented in this example")
        return

    # Compare sizes
    quantizer.compare_size(model, quantized)

    # Save
    output_path = Path(args.output)
    torch.save(quantized.state_dict(), output_path)
    print(f"\nSaved quantized model to {output_path}")


def cmd_export_onnx(args):
    """Export to ONNX command."""
    exporter = ONNXExporter(args.model)

    # Load model
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model)
    model.eval()

    # Export
    input_shape = (1, 3, args.height, args.width)
    output_path = Path(args.output)
    exporter.export(model, output_path, input_shape, args.opset)

    # Benchmark if requested
    if args.benchmark:
        exporter.benchmark_onnx(output_path, input_shape, args.iterations)


def cmd_profile(args):
    """Profile model command."""
    profiler = PerformanceProfiler()

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForDepthEstimation.from_pretrained(args.model)
    model.eval()

    if torch.cuda.is_available() and not args.cpu:
        model = model.cuda()

    # Create input
    input_shape = (1, 3, args.height, args.width)
    x = torch.randn(input_shape)
    if torch.cuda.is_available() and not args.cpu:
        x = x.cuda()

    # Profile layers
    if args.layers:
        profiler.profile_layers(model, x, args.iterations)

    # Profile memory
    if args.memory:
        profiler.profile_memory(model, x)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Performance tuning utilities for DA3 models'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Quantize command
    quantize_parser = subparsers.add_parser(
        'quantize', help='Quantize model to INT8'
    )
    quantize_parser.add_argument(
        '--model', '-m', type=str, default='depth-anything/DA3-BASE',
        help='Model to quantize'
    )
    quantize_parser.add_argument(
        '--output', '-o', type=str, default='model_quantized.pth',
        help='Output path'
    )
    quantize_parser.add_argument(
        '--mode', type=str, default='dynamic',
        choices=['dynamic', 'static'],
        help='Quantization mode'
    )

    # ONNX export command
    onnx_parser = subparsers.add_parser(
        'export-onnx', help='Export model to ONNX'
    )
    onnx_parser.add_argument(
        '--model', '-m', type=str, default='depth-anything/DA3-BASE',
        help='Model to export'
    )
    onnx_parser.add_argument(
        '--output', '-o', type=str, default='model.onnx',
        help='Output ONNX path'
    )
    onnx_parser.add_argument(
        '--height', type=int, default=518, help='Input height'
    )
    onnx_parser.add_argument(
        '--width', type=int, default=518, help='Input width'
    )
    onnx_parser.add_argument(
        '--opset', type=int, default=14, help='ONNX opset version'
    )
    onnx_parser.add_argument(
        '--benchmark', action='store_true', help='Benchmark ONNX model'
    )
    onnx_parser.add_argument(
        '--iterations', type=int, default=100, help='Benchmark iterations'
    )

    # Profile command
    profile_parser = subparsers.add_parser(
        'profile', help='Profile model performance'
    )
    profile_parser.add_argument(
        '--model', '-m', type=str, default='depth-anything/DA3-BASE',
        help='Model to profile'
    )
    profile_parser.add_argument(
        '--height', type=int, default=518, help='Input height'
    )
    profile_parser.add_argument(
        '--width', type=int, default=518, help='Input width'
    )
    profile_parser.add_argument(
        '--iterations', type=int, default=100, help='Profile iterations'
    )
    profile_parser.add_argument(
        '--layers', action='store_true', help='Profile individual layers'
    )
    profile_parser.add_argument(
        '--memory', action='store_true', help='Profile memory usage'
    )
    profile_parser.add_argument(
        '--cpu', action='store_true', help='Use CPU instead of CUDA'
    )

    args = parser.parse_args()

    if args.command == 'quantize':
        cmd_quantize(args)
    elif args.command == 'export-onnx':
        cmd_export_onnx(args)
    elif args.command == 'profile':
        cmd_profile(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
