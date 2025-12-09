"""
GPU-accelerated utilities for high-performance depth processing.

This module provides CUDA-optimized operations for depth map upsampling,
image preprocessing, and other GPU-accelerated operations to achieve
real-time performance (>30 FPS) on NVIDIA Jetson platforms.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class GPUDepthUpsampler:
    """
    GPU-accelerated depth map upsampling for real-time performance.

    Provides multiple upsampling modes optimized for Jetson platforms:
    - bilinear: Fastest, good for smooth depth maps
    - bicubic: Better quality, slightly slower
    - nearest: Fastest, preserves sharp edges but blocky

    All operations are performed on GPU to minimize CPU-GPU transfers.
    """

    def __init__(self, mode: str = "bilinear", device: str = "cuda"):
        """
        Initialize GPU upsampler.

        Args:
            mode: Interpolation mode ('bilinear', 'bicubic', 'nearest')
            device: Compute device ('cuda' or 'cpu')
        """
        self.mode = mode
        self.device = device

        if mode not in ["bilinear", "bicubic", "nearest"]:
            raise ValueError(
                f"Invalid mode '{mode}'. "
                f"Must be 'bilinear', 'bicubic', or 'nearest'"
            )

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"GPU upsampler initialized: mode={mode}, device={self.device}")

    def upsample_tensor(
        self, tensor: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Upsample a tensor on GPU.

        Args:
            tensor: Input tensor (B, C, H, W) or (H, W) on GPU
            target_size: Target size as (height, width)

        Returns:
            Upsampled tensor on same device
        """
        # Validate inputs
        if tensor is None:
            raise ValueError("Tensor cannot be None")

        if tensor.numel() == 0:
            raise ValueError("Tensor is empty")

        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"Invalid target size: {target_size}")

        # Ensure 4D tensor (B, C, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {tensor.ndim}D")

        # Perform interpolation
        upsampled = F.interpolate(
            tensor,
            size=target_size,
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        )

        return upsampled

    def upsample_numpy(
        self, array: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Upsample a numpy array using GPU acceleration.

        Args:
            array: Input numpy array (H, W) or (H, W, C)
            target_size: Target size as (height, width)

        Returns:
            Upsampled numpy array
        """
        # Validate inputs
        if array is None:
            raise ValueError("Array cannot be None")

        if array.size == 0:
            raise ValueError("Array is empty")

        if not np.isfinite(array).all():
            raise ValueError("Array contains NaN or infinite values")

        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"Invalid target size: {target_size}")

        # Convert to tensor and move to GPU
        tensor = torch.from_numpy(array).to(self.device)

        # Handle different input shapes
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            single_channel = True
        elif tensor.ndim == 3:
            # (H, W, C) -> (1, C, H, W)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            single_channel = False
        else:
            raise ValueError(f"Invalid array shape: {array.shape}")

        # Upsample
        upsampled = F.interpolate(
            tensor,
            size=target_size,
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        )

        # Convert back to numpy
        if single_channel:
            result = upsampled.squeeze(0).squeeze(0).cpu().numpy()
        else:
            result = upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return result.astype(array.dtype)


class GPUImagePreprocessor:
    """
    GPU-accelerated image preprocessing for depth estimation.

    Handles resizing, normalization, and format conversions on GPU
    to minimize CPU-GPU transfer overhead.
    """

    def __init__(self, target_size: Tuple[int, int] = (384, 384), device: str = "cuda"):
        """
        Initialize GPU preprocessor.

        Args:
            target_size: Target size for model input as (height, width)
            device: Compute device ('cuda' or 'cpu')
        """
        self.target_size = target_size
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

    def preprocess(
        self, image: np.ndarray, return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model input on GPU.

        Args:
            image: Input RGB image as numpy array (H, W, 3)
            return_tensor: If True, return torch.Tensor, else numpy array

        Returns:
            Preprocessed image (1, 3, H, W) tensor or (H, W, 3) array
        """
        # Convert to tensor and move to GPU
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).to(self.device)
        else:
            tensor = image.to(self.device)

        # Ensure correct dtype (float32 for model input)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0

        # Handle shape: (H, W, 3) -> (1, 3, H, W)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        # Resize to target size
        if tensor.shape[2:] != self.target_size:
            tensor = F.interpolate(
                tensor, size=self.target_size, mode="bilinear", align_corners=False
            )

        if return_tensor:
            return tensor
        else:
            # Convert back to numpy
            return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


class CUDAStreamManager:
    """
    Manages CUDA streams for overlapping computation and data transfer.

    Enables pipeline parallelism to hide latency:
    - Stream 0: Image acquisition and preprocessing
    - Stream 1: Model inference
    - Stream 2: Postprocessing and publishing
    """

    def __init__(self, num_streams: int = 3):
        """
        Initialize CUDA stream manager.

        Args:
            num_streams: Number of CUDA streams to create
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, stream management disabled")
            self.streams = None
            self.enabled = False
            return

        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.enabled = True
        logger.info(f"CUDA stream manager initialized with {num_streams} streams")

    def get_stream(self, idx: int) -> Optional[torch.cuda.Stream]:
        """
        Get CUDA stream by index.

        Args:
            idx: Stream index

        Returns:
            CUDA stream or None if not available
        """
        if not self.enabled or self.streams is None:
            return None

        if idx < 0 or idx >= len(self.streams):
            raise ValueError(f"Invalid stream index: {idx}")

        return self.streams[idx]

    def synchronize_all(self):
        """Synchronize all streams."""
        if self.enabled and self.streams is not None:
            for stream in self.streams:
                stream.synchronize()

    def cleanup(self):
        """Clean up CUDA streams."""
        if self.enabled and self.streams is not None:
            self.synchronize_all()
            self.streams = None
            self.enabled = False
            logger.info("CUDA streams cleaned up")


def tensor_to_numpy_gpu(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert GPU tensor to numpy array with minimal overhead.

    Args:
        tensor: Input tensor on GPU

    Returns:
        Numpy array on CPU
    """
    return tensor.detach().cpu().numpy()


def numpy_to_tensor_gpu(
    array: np.ndarray, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert numpy array to GPU tensor with minimal overhead.

    Args:
        array: Input numpy array
        device: Target device
        dtype: Target dtype

    Returns:
        Tensor on specified device
    """
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def pinned_numpy_array(shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """
    Create a pinned (page-locked) numpy array for faster CPU-GPU transfers.

    Args:
        shape: Array shape
        dtype: Array dtype

    Returns:
        Pinned numpy array
    """
    if not torch.cuda.is_available():
        return np.zeros(shape, dtype=dtype)

    # Map numpy dtype to torch dtype
    dtype_map = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.uint8: torch.uint8,
    }

    torch_dtype = dtype_map.get(dtype, torch.float32)

    # Create tensor with pinned memory
    tensor = torch.zeros(shape, dtype=torch_dtype, pin_memory=True)
    # Get numpy view
    return tensor.numpy()


class GPUMemoryMonitor:
    """Monitor GPU memory usage for performance tuning."""

    @staticmethod
    def get_memory_stats() -> dict:
        """
        Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats in MB
        """
        if not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "total_mb": 0.0,
            }

        # Use current device instead of hardcoded device 0
        device_id = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**2)

        # Get total memory for current device
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**2)
        free = total - allocated

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free,
            "total_mb": total,
        }

    @staticmethod
    def clear_cache():
        """Clear CUDA cache to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
