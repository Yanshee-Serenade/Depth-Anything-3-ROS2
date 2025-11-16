#!/usr/bin/env python3
"""
Multi-Camera Synchronized Depth Estimation

Demonstrates synchronized depth estimation from multiple cameras:
- Subscribes to multiple depth topics
- Synchronizes frames by timestamp
- Merges depth maps for wider field of view
- Saves synchronized depth sequences

Requirements:
    ROS2 Humble
    pip install numpy opencv-python

Usage:
    # Terminal 1: Launch multi-camera depth estimation
    ros2 launch depth_anything_3_ros2 multi_camera.launch.py \
        camera_namespaces:=cam_left,cam_right,cam_center \
        image_topics:=/cam_left/image_raw,/cam_right/image_raw,/cam_center/image_raw

    # Terminal 2: Run synchronization
    python3 multi_camera_sync.py \
        --cameras cam_left cam_right cam_center \
        --output synchronized_depth/ \
        --sync-threshold 0.05
"""

import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


class MultiCameraSync(Node):
    """Synchronize depth estimation from multiple cameras."""

    def __init__(
        self,
        camera_names: List[str],
        output_dir: Path,
        sync_threshold: float = 0.05,
        buffer_size: int = 30,
        save_interval: int = 10,
    ):
        """
        Initialize multi-camera synchronizer.

        Args:
            camera_names: List of camera namespace prefixes
            output_dir: Directory to save synchronized outputs
            sync_threshold: Max time difference for sync in seconds
            buffer_size: Number of frames to buffer per camera
            save_interval: Save every N synchronized frames
        """
        super().__init__('multi_camera_sync')

        self.camera_names = camera_names
        self.output_dir = output_dir
        self.sync_threshold = sync_threshold
        self.save_interval = save_interval
        self.bridge = CvBridge()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frame buffers for each camera
        self.depth_buffers: Dict[str, deque] = {
            cam: deque(maxlen=buffer_size) for cam in camera_names
        }
        self.colored_buffers: Dict[str, deque] = {
            cam: deque(maxlen=buffer_size) for cam in camera_names
        }

        # Statistics
        self.frame_count = 0
        self.synced_count = 0
        self.start_time = time.time()

        # Create subscribers for each camera
        self.depth_subs = []
        self.colored_subs = []

        for cam in camera_names:
            # Depth topic
            depth_sub = self.create_subscription(
                Image,
                f'/{cam}/depth_anything_3/depth',
                lambda msg, c=cam: self.depth_callback(msg, c),
                10
            )
            self.depth_subs.append(depth_sub)

            # Colored depth topic
            colored_sub = self.create_subscription(
                Image,
                f'/{cam}/depth_anything_3/depth_colored',
                lambda msg, c=cam: self.colored_callback(msg, c),
                10
            )
            self.colored_subs.append(colored_sub)

            self.get_logger().info(f"Subscribed to /{cam}/depth_anything_3/*")

        # Timer for synchronization attempts
        self.sync_timer = self.create_timer(0.1, self.sync_callback)

        # Timer for statistics
        self.stats_timer = self.create_timer(5.0, self.print_statistics)

        self.get_logger().info(
            f"Multi-camera sync initialized for {len(camera_names)} cameras"
        )
        self.get_logger().info(f"Sync threshold: {sync_threshold}s")
        self.get_logger().info(f"Output directory: {output_dir}")

    def depth_callback(self, msg: Image, camera_name: str):
        """Callback for depth images."""
        self.depth_buffers[camera_name].append(msg)

    def colored_callback(self, msg: Image, camera_name: str):
        """Callback for colored depth images."""
        self.colored_buffers[camera_name].append(msg)

    def get_timestamp(self, msg: Image) -> float:
        """Extract timestamp from message as float seconds."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def find_synchronized_frames(
        self,
    ) -> Optional[Dict[str, Tuple[Image, Image]]]:
        """
        Find synchronized frames across all cameras.

        Returns:
            Dictionary mapping camera names to (depth, colored) tuples,
            or None if no synchronized set found
        """
        # Check if all buffers have data
        if any(
            len(self.depth_buffers[cam]) == 0
            or len(self.colored_buffers[cam]) == 0
            for cam in self.camera_names
        ):
            return None

        # Get most recent timestamps from each camera
        latest_times = {}
        for cam in self.camera_names:
            depth_time = self.get_timestamp(self.depth_buffers[cam][-1])
            colored_time = self.get_timestamp(self.colored_buffers[cam][-1])
            latest_times[cam] = max(depth_time, colored_time)

        # Find reference time (median of latest times)
        ref_time = np.median(list(latest_times.values()))

        # Try to find matching frames for all cameras
        matched_frames = {}

        for cam in self.camera_names:
            # Find closest depth frame
            depth_frame = None
            min_depth_diff = float('inf')

            for msg in self.depth_buffers[cam]:
                time_diff = abs(self.get_timestamp(msg) - ref_time)
                if time_diff < min_depth_diff:
                    min_depth_diff = time_diff
                    depth_frame = msg

            # Find closest colored frame
            colored_frame = None
            min_colored_diff = float('inf')

            for msg in self.colored_buffers[cam]:
                time_diff = abs(self.get_timestamp(msg) - ref_time)
                if time_diff < min_colored_diff:
                    min_colored_diff = time_diff
                    colored_frame = msg

            # Check if within threshold
            if (
                min_depth_diff > self.sync_threshold
                or min_colored_diff > self.sync_threshold
            ):
                return None

            matched_frames[cam] = (depth_frame, colored_frame)

        return matched_frames

    def sync_callback(self):
        """Attempt to synchronize and process frames."""
        matched = self.find_synchronized_frames()

        if matched is None:
            return

        self.synced_count += 1

        # Save frames if at save interval
        if self.synced_count % self.save_interval == 0:
            self.save_synchronized_frames(matched)

        # Clear processed frames from buffers
        self.clear_old_frames(matched)

    def save_synchronized_frames(
        self, frames: Dict[str, Tuple[Image, Image]]
    ):
        """Save synchronized frame set."""
        frame_dir = self.output_dir / f"frame_{self.synced_count:06d}"
        frame_dir.mkdir(exist_ok=True)

        for cam, (depth_msg, colored_msg) in frames.items():
            # Convert depth to numpy
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            )

            # Convert colored to numpy
            colored = self.bridge.imgmsg_to_cv2(
                colored_msg, desired_encoding='bgr8'
            )

            # Save depth as numpy array
            depth_path = frame_dir / f"{cam}_depth.npy"
            np.save(depth_path, depth)

            # Save colored visualization
            colored_path = frame_dir / f"{cam}_depth_colored.png"
            cv2.imwrite(str(colored_path), colored)

        self.get_logger().info(
            f"Saved synchronized frame {self.synced_count} to {frame_dir}"
        )

    def clear_old_frames(self, matched: Dict[str, Tuple[Image, Image]]):
        """Remove processed frames from buffers."""
        for cam, (depth_msg, colored_msg) in matched.items():
            depth_time = self.get_timestamp(depth_msg)
            colored_time = self.get_timestamp(colored_msg)

            # Remove depth frames older than matched
            while (
                self.depth_buffers[cam]
                and self.get_timestamp(self.depth_buffers[cam][0])
                <= depth_time
            ):
                self.depth_buffers[cam].popleft()

            # Remove colored frames older than matched
            while (
                self.colored_buffers[cam]
                and self.get_timestamp(self.colored_buffers[cam][0])
                <= colored_time
            ):
                self.colored_buffers[cam].popleft()

    def print_statistics(self):
        """Print synchronization statistics."""
        elapsed = time.time() - self.start_time
        sync_rate = self.synced_count / elapsed if elapsed > 0 else 0

        # Buffer status
        buffer_info = []
        for cam in self.camera_names:
            depth_len = len(self.depth_buffers[cam])
            colored_len = len(self.colored_buffers[cam])
            buffer_info.append(f"{cam}: D={depth_len}, C={colored_len}")

        self.get_logger().info(
            f"Synced frames: {self.synced_count} "
            f"({sync_rate:.2f} Hz) | "
            f"Buffers: {', '.join(buffer_info)}"
        )


def create_panoramic_depth(
    depth_maps: List[np.ndarray],
    camera_positions: List[Tuple[float, float]],
    output_width: int = 1920,
    output_height: int = 480,
) -> np.ndarray:
    """
    Create panoramic depth map from multiple cameras.

    Args:
        depth_maps: List of depth maps from each camera
        camera_positions: List of (horizontal_angle, vertical_angle) in degrees
        output_width: Width of panoramic output
        output_height: Height of panoramic output

    Returns:
        Panoramic depth map
    """
    panorama = np.zeros((output_height, output_width), dtype=np.float32)
    weight_map = np.zeros((output_height, output_width), dtype=np.float32)

    for depth, (h_angle, v_angle) in zip(depth_maps, camera_positions):
        # Calculate placement in panorama
        h_offset = int((h_angle / 360.0) * output_width)

        # Simple blending (could be improved with proper projection)
        h, w = depth.shape
        x_start = max(0, h_offset)
        x_end = min(output_width, h_offset + w)
        y_start = (output_height - h) // 2
        y_end = y_start + h

        # Extract valid region
        depth_x_start = max(0, -h_offset)
        depth_x_end = depth_x_start + (x_end - x_start)

        # Blend with existing panorama
        panorama[y_start:y_end, x_start:x_end] += (
            depth[:, depth_x_start:depth_x_end]
        )
        weight_map[y_start:y_end, x_start:x_end] += 1.0

    # Average overlapping regions
    mask = weight_map > 0
    panorama[mask] /= weight_map[mask]

    return panorama


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Synchronize multi-camera depth estimation'
    )
    parser.add_argument(
        '--cameras', '-c', nargs='+', required=True,
        help='Camera namespace prefixes (e.g., cam_left cam_right)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='synchronized_depth',
        help='Output directory (default: synchronized_depth)'
    )
    parser.add_argument(
        '--sync-threshold', type=float, default=0.05,
        help='Max time difference for synchronization in seconds (default: 0.05)'
    )
    parser.add_argument(
        '--buffer-size', type=int, default=30,
        help='Frame buffer size per camera (default: 30)'
    )
    parser.add_argument(
        '--save-interval', type=int, default=10,
        help='Save every N synchronized frames (default: 10)'
    )

    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    # Create output directory
    output_dir = Path(args.output)

    # Create node
    node = MultiCameraSync(
        camera_names=args.cameras,
        output_dir=output_dir,
        sync_threshold=args.sync_threshold,
        buffer_size=args.buffer_size,
        save_interval=args.save_interval,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.get_logger().info(
            f"Total synchronized frames: {node.synced_count}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
