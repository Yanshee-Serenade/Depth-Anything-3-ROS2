"""
Unit tests for DA3 inference wrapper.

Tests the core inference functionality including model loading,
inference execution, and error handling.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock


class TestDA3InferenceWrapper(unittest.TestCase):
    """Test cases for DA3InferenceWrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @patch("depth_anything_3_ros2.da3_inference.torch")
    @patch("depth_anything_3.api.DepthAnything3")
    def test_device_selection_cuda_available(self, mock_da3, mock_torch):
        """Test device selection when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GPU"

        from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper

        mock_model = MagicMock()
        mock_da3.from_pretrained.return_value = mock_model

        wrapper = DA3InferenceWrapper(model_name="test-model", device="cuda")

        self.assertEqual(wrapper.device, "cuda")
        mock_torch.cuda.is_available.assert_called_once()

    @patch("depth_anything_3_ros2.da3_inference.torch")
    @patch("depth_anything_3.api.DepthAnything3")
    def test_device_selection_cuda_unavailable(self, mock_da3, mock_torch):
        """Test device fallback to CPU when CUDA is unavailable."""
        mock_torch.cuda.is_available.return_value = False

        from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper

        mock_model = MagicMock()
        mock_da3.from_pretrained.return_value = mock_model

        wrapper = DA3InferenceWrapper(model_name="test-model", device="cuda")

        self.assertEqual(wrapper.device, "cpu")

    @patch("depth_anything_3_ros2.da3_inference.torch")
    @patch("depth_anything_3.api.DepthAnything3")
    def test_inference_with_valid_image(self, mock_da3, mock_torch):
        """Test inference with valid input image."""
        from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper

        # Setup mock
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_da3.from_pretrained.return_value = mock_model

        # Mock prediction result
        mock_prediction = MagicMock()
        mock_prediction.depth = [np.random.rand(480, 640).astype(np.float32)]
        mock_prediction.conf = [np.random.rand(480, 640).astype(np.float32)]
        mock_model.inference.return_value = mock_prediction

        wrapper = DA3InferenceWrapper(model_name="test-model", device="cpu")

        # Test inference
        result = wrapper.inference(self.test_image, return_confidence=True)

        self.assertIn("depth", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["depth"].shape, (480, 640))
        self.assertEqual(result["confidence"].shape, (480, 640))

    def test_inference_invalid_input_type(self):
        """Test inference with invalid input type."""
        with patch("depth_anything_3_ros2.da3_inference.torch") as mock_torch:
            with patch("depth_anything_3.api.DepthAnything3") as mock_da3:
                from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper

                mock_torch.cuda.is_available.return_value = False
                mock_model = MagicMock()
                mock_da3.from_pretrained.return_value = mock_model

                wrapper = DA3InferenceWrapper(model_name="test-model")

                # Test that invalid input types raise ValueError
                with self.assertRaises(ValueError):
                    wrapper.inference("not an array")

    @patch("depth_anything_3_ros2.da3_inference.torch")
    @patch("depth_anything_3.api.DepthAnything3")
    def test_gpu_memory_usage_cuda(self, mock_da3, mock_torch):
        """Test GPU memory usage reporting with CUDA."""
        from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GPU"
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100
        mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 150

        mock_model = MagicMock()
        mock_da3.from_pretrained.return_value = mock_model

        wrapper = DA3InferenceWrapper(model_name="test-model", device="cuda")
        mem_usage = wrapper.get_gpu_memory_usage()

        self.assertIsNotNone(mem_usage)
        self.assertIn("allocated_mb", mem_usage)
        self.assertIn("reserved_mb", mem_usage)
        self.assertAlmostEqual(mem_usage["allocated_mb"], 100.0, places=1)
        self.assertAlmostEqual(mem_usage["reserved_mb"], 150.0, places=1)

    @patch("depth_anything_3_ros2.da3_inference.torch")
    @patch("depth_anything_3.api.DepthAnything3")
    def test_gpu_memory_usage_cpu(self, mock_da3, mock_torch):
        """Test GPU memory usage reporting with CPU."""
        from depth_anything_3_ros2.da3_inference import DA3InferenceWrapper

        mock_torch.cuda.is_available.return_value = False

        mock_model = MagicMock()
        mock_da3.from_pretrained.return_value = mock_model

        wrapper = DA3InferenceWrapper(model_name="test-model", device="cpu")
        mem_usage = wrapper.get_gpu_memory_usage()

        self.assertIsNone(mem_usage)


if __name__ == "__main__":
    unittest.main()
