#!/usr/bin/env python3
"""
Profile ROS2 node performance with various configurations.

This script runs the depth estimation node with different configurations
and measures memory usage, CPU usage, and latency.
"""

import argparse
import subprocess
import time
import json
import psutil
from pathlib import Path
from typing import Dict, List
import signal
import sys


class NodeProfiler:
    """Profiler for ROS2 node performance."""

    def __init__(self, output_file: Path):
        """
        Initialize profiler.

        Args:
            output_file: Path to save profiling results
        """
        self.output_file = output_file
        self.results = []
        self.running_processes = []

    def profile_configuration(
        self,
        model_name: str,
        device: str,
        image_size: tuple = (640, 480),
        duration: int = 30
    ) -> Dict:
        """
        Profile a specific configuration.

        Args:
            model_name: DA3 model name
            device: cuda or cpu
            image_size: Image resolution as (width, height)
            duration: Profile duration in seconds

        Returns:
            Dictionary with profiling results
        """
        print(f"\nProfiling: {model_name} on {device} "
              f"@ {image_size[0]}x{image_size[1]}")
        print(f"Duration: {duration}s")

        # Build launch command
        cmd = [
            'ros2', 'run', 'depth_anything_3_ros2', 'depth_anything_3_node',
            '--ros-args',
            '-p', f'model_name:={model_name}',
            '-p', f'device:={device}',
            '-p', 'log_inference_time:=true'
        ]

        # Start node
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.running_processes.append(process)

        # Wait for node to initialize
        time.sleep(5)

        # Monitor resource usage
        samples = []
        start_time = time.time()

        try:
            node_process = psutil.Process(process.pid)

            while time.time() - start_time < duration:
                try:
                    # Get resource usage
                    cpu_percent = node_process.cpu_percent(interval=1.0)
                    memory_info = node_process.memory_info()
                    memory_mb = memory_info.rss / (1024 ** 2)

                    samples.append({
                        'timestamp': time.time() - start_time,
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb
                    })

                    print(f"  Sample: CPU={cpu_percent:.1f}% "
                          f"RAM={memory_mb:.1f}MB", end='\r')

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        finally:
            # Stop node
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            self.running_processes.remove(process)

        # Calculate statistics
        if samples:
            cpu_values = [s['cpu_percent'] for s in samples]
            memory_values = [s['memory_mb'] for s in samples]

            result = {
                'model_name': model_name,
                'device': device,
                'image_size': image_size,
                'duration': duration,
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                'max_cpu_percent': max(cpu_values),
                'avg_memory_mb': sum(memory_values) / len(memory_values),
                'max_memory_mb': max(memory_values),
                'num_samples': len(samples)
            }

            print(f"\n  Results: CPU={result['avg_cpu_percent']:.1f}% "
                  f"RAM={result['avg_memory_mb']:.1f}MB")

            return result
        else:
            return None

    def run_profiles(
        self,
        models: List[str],
        devices: List[str],
        duration: int = 30
    ):
        """
        Run profiling for multiple configurations.

        Args:
            models: List of model names
            devices: List of devices (cuda/cpu)
            duration: Profile duration per configuration
        """
        total = len(models) * len(devices)
        current = 0

        for model in models:
            for device in devices:
                current += 1
                print(f"\n[{current}/{total}] Profiling configuration...")

                result = self.profile_configuration(
                    model_name=model,
                    device=device,
                    duration=duration
                )

                if result:
                    self.results.append(result)

        # Save results
        self.save_results()

    def save_results(self):
        """Save profiling results to JSON."""
        with open(self.output_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'results': self.results
            }, f, indent=2)

        print(f"\nResults saved to: {self.output_file}")

    def print_summary(self):
        """Print summary of profiling results."""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        print(f"{'Model':<30} {'Device':<8} {'CPU %':<10} {'RAM (MB)':<12}")
        print("-"*80)

        for r in self.results:
            print(f"{r['model_name']:<30} {r['device']:<8} "
                  f"{r['avg_cpu_percent']:<10.1f} "
                  f"{r['avg_memory_mb']:<12.1f}")

        print("="*80)

    def cleanup(self):
        """Cleanup running processes."""
        for process in self.running_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    process.kill()
                except Exception:
                    pass


def signal_handler(signum, frame):
    """Handle termination signals."""
    print("\nInterrupted! Cleaning up...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Profile ROS2 node performance'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='depth-anything/DA3-BASE',
        help='Comma-separated list of models to profile'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='cuda,cpu',
        help='Comma-separated list of devices (cuda/cpu)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Profile duration per configuration (seconds)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('profile_results.json'),
        help='Output file for results'
    )

    args = parser.parse_args()

    # Parse arguments
    models = [m.strip() for m in args.models.split(',')]
    devices = [d.strip() for d in args.devices.split(',')]

    print("="*80)
    print("ROS2 NODE PERFORMANCE PROFILER")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Devices: {', '.join(devices)}")
    print(f"Duration per config: {args.duration}s")
    print("="*80)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run profiler
    profiler = NodeProfiler(args.output)

    try:
        profiler.run_profiles(models, devices, args.duration)
        profiler.print_summary()
    finally:
        profiler.cleanup()


if __name__ == '__main__':
    main()
