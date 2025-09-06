"""
Pytest configuration and shared fixtures for HAILO RPI5 tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_hailo_device():
    """Mock HAILO device for testing."""
    device = Mock()
    device.device_id = "test_device_123"
    device.device_info = Mock()
    device.device_info.device_architecture = "hailo8"
    device.configure = Mock()
    device.activate = Mock()
    device.deactivate = Mock()
    device.release = Mock()
    return device


@pytest.fixture
def mock_vstream():
    """Mock VStream for testing."""
    vstream = Mock()
    vstream.name = "test_vstream"
    vstream.info = Mock()
    vstream.info.shape = (224, 224, 3)
    vstream.info.format = Mock()
    vstream.info.format.type = "UINT8"
    return vstream


@pytest.fixture
def mock_inference_engine(mock_hailo_device, mock_vstream):
    """Mock inference engine with device and vstream."""
    from src.core.inference_engine import HailoInferenceEngine
    
    with pytest.mock.patch('src.core.inference_engine.HAILO_AVAILABLE', True):
        engine = HailoInferenceEngine()
        engine.device = mock_hailo_device
        engine.input_vstreams = [mock_vstream]
        engine.output_vstreams = [mock_vstream]
        engine.network_group = Mock()
        return engine


@pytest.fixture
def sample_image():
    """Sample image for testing."""
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_model_path():
    """Sample model path for testing."""
    return "/path/to/test_model.hef"


@pytest.fixture
def mock_cv2_image():
    """Mock OpenCV image for testing."""
    return np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_content = """
device:
    timeout: 5000
    
preprocessing:
    resize_method: "bilinear"
    normalize: true
    
postprocessing:
    confidence_threshold: 0.5
    nms_threshold: 0.4
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)
