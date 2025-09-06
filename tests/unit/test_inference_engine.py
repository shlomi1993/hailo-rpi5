"""
Unit tests for HailoInferenceEngine.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.inference_engine import HailoInferenceEngine


class TestHailoInferenceEngine:
    """Test cases for HailoInferenceEngine."""

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    def test_init_with_hailo_available(self):
        """Test initialization when HAILO is available."""
        engine = HailoInferenceEngine()
        assert engine is not None
        assert engine.device is None
        assert engine.network_group is None

    @patch('src.core.inference_engine.HAILO_AVAILABLE', False)
    def test_init_without_hailo(self):
        """Test initialization when HAILO is not available."""
        with pytest.raises(RuntimeError, match="HAILO platform not available"):
            HailoInferenceEngine()

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    @patch('src.core.inference_engine.HailoDeviceManager')
    def test_load_model_success(self, mock_device_manager):
        """Test successful model loading."""
        # Setup mocks
        mock_device = Mock()
        mock_network_group = Mock()
        mock_device_manager.return_value.get_device.return_value = mock_device
        mock_device.configure.return_value = [mock_network_group]

        # Test
        engine = HailoInferenceEngine()
        result = engine.load_model("/path/to/model.hef")

        # Assertions
        assert result is True
        assert engine.device == mock_device
        assert engine.network_group == mock_network_group

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    @patch('src.core.inference_engine.HailoDeviceManager')
    def test_load_model_device_not_found(self, mock_device_manager):
        """Test model loading when device is not found."""
        # Setup mocks
        mock_device_manager.return_value.get_device.return_value = None

        # Test
        engine = HailoInferenceEngine()
        result = engine.load_model("/path/to/model.hef")

        # Assertions
        assert result is False
        assert engine.device is None

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    def test_preprocess_image(self):
        """Test image preprocessing."""
        engine = HailoInferenceEngine()
        
        # Create a sample image
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        target_shape = (224, 224, 3)
        
        # Test preprocessing
        preprocessed = engine.preprocess(image, target_shape)
        
        # Assertions
        assert preprocessed.shape == target_shape
        assert preprocessed.dtype == np.uint8

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    def test_run_inference_not_loaded(self):
        """Test inference when model is not loaded."""
        engine = HailoInferenceEngine()
        image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.run_inference(image)

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    @patch('src.core.inference_engine.HailoDeviceManager')
    def test_run_inference_success(self, mock_device_manager):
        """Test successful inference run."""
        # Setup mocks
        mock_device = Mock()
        mock_network_group = Mock()
        mock_input_vstream = Mock()
        mock_output_vstream = Mock()
        
        mock_device_manager.return_value.get_device.return_value = mock_device
        mock_device.configure.return_value = [mock_network_group]
        
        # Mock VStreams
        mock_network_group.create_input_vstreams.return_value = [mock_input_vstream]
        mock_network_group.create_output_vstreams.return_value = [mock_output_vstream]
        
        # Mock inference results
        mock_output_data = np.random.rand(1000).astype(np.float32)
        mock_output_vstream.recv.return_value = mock_output_data
        
        # Test
        engine = HailoInferenceEngine()
        engine.load_model("/path/to/model.hef")
        
        image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        results = engine.run_inference(image)
        
        # Assertions
        assert results is not None
        assert len(results) > 0

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    def test_cleanup(self):
        """Test cleanup functionality."""
        engine = HailoInferenceEngine()
        engine.device = Mock()
        engine.network_group = Mock()
        engine.input_vstreams = [Mock()]
        engine.output_vstreams = [Mock()]
        
        # Test cleanup
        engine.cleanup()
        
        # Assertions
        assert engine.device is None
        assert engine.network_group is None
        assert engine.input_vstreams == []
        assert engine.output_vstreams == []

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    def test_context_manager(self):
        """Test context manager functionality."""
        engine = HailoInferenceEngine()
        engine.cleanup = Mock()
        
        with engine:
            pass
        
        # Should call cleanup when exiting context
        engine.cleanup.assert_called_once()

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    def test_get_model_info_no_model(self):
        """Test getting model info when no model is loaded."""
        engine = HailoInferenceEngine()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.get_model_info()

    @patch('src.core.inference_engine.HAILO_AVAILABLE', True)
    @patch('src.core.inference_engine.HailoDeviceManager')
    def test_get_model_info_with_model(self, mock_device_manager):
        """Test getting model info when model is loaded."""
        # Setup mocks
        mock_device = Mock()
        mock_network_group = Mock()
        mock_network_group.get_network_infos.return_value = [{'name': 'test_network'}]
        
        mock_device_manager.return_value.get_device.return_value = mock_device
        mock_device.configure.return_value = [mock_network_group]
        
        # Test
        engine = HailoInferenceEngine()
        engine.load_model("/path/to/model.hef")
        
        info = engine.get_model_info()
        
        # Assertions
        assert info is not None
        assert 'network_infos' in info
