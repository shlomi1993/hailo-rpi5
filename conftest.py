# pytest configuration file
# This file provides additional pytest configuration beyond pyproject.toml

import os
import sys
import pytest
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Environment setup for tests
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set test environment
    os.environ["HAILO_TEST_MODE"] = "1"
    os.environ["HAILO_LOG_LEVEL"] = "DEBUG"
    
    # Disable GPU/device access during testing unless explicitly needed
    if not config.getoption("--device-tests"):
        os.environ["HAILO_MOCK_DEVICE"] = "1"

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--device-tests",
        action="store_true",
        default=False,
        help="Run tests that require actual HAILO device"
    )
    parser.addoption(
        "--model-tests",
        action="store_true", 
        default=False,
        help="Run tests that require model files"
    )
    parser.addoption(
        "--integration-tests",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--performance-tests",
        action="store_true",
        default=False,
        help="Run performance/benchmark tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    # Skip device tests unless explicitly requested
    if not config.getoption("--device-tests"):
        skip_device = pytest.mark.skip(reason="need --device-tests option to run")
        for item in items:
            if "device" in item.keywords:
                item.add_marker(skip_device)
    
    # Skip model tests unless explicitly requested
    if not config.getoption("--model-tests"):
        skip_model = pytest.mark.skip(reason="need --model-tests option to run")
        for item in items:
            if "model" in item.keywords:
                item.add_marker(skip_model)
    
    # Skip integration tests unless explicitly requested
    if not config.getoption("--integration-tests"):
        skip_integration = pytest.mark.skip(reason="need --integration-tests option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    # Skip performance tests unless explicitly requested
    if not config.getoption("--performance-tests"):
        skip_performance = pytest.mark.skip(reason="need --performance-tests option to run")
        for item in items:
            if "slow" in item.keywords or "performance" in item.keywords:
                item.add_marker(skip_performance)

@pytest.fixture(scope="session")
def project_root():
    """Provide the project root directory."""
    return Path(__file__).parent.absolute()

@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Provide the test data directory."""
    return project_root / "tests" / "data"

@pytest.fixture(scope="session")
def models_dir(project_root):
    """Provide the models directory."""
    return project_root / "models"

@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Provide a temporary directory for tests."""
    return tmp_path_factory.mktemp("hailo_tests")

@pytest.fixture
def mock_device():
    """Provide a mock HAILO device for testing."""
    from unittest.mock import MagicMock
    
    device = MagicMock()
    device.id = "mock_device_0"
    device.is_connected.return_value = True
    device.get_extended_device_information.return_value = {
        "device_id": "mock_device_0",
        "serial_number": "MOCK12345",
        "part_number": "HAILO8",
        "product_name": "Hailo-8"
    }
    
    return device

@pytest.fixture
def sample_image():
    """Provide a sample test image."""
    import numpy as np
    from PIL import Image
    
    # Create a simple test image (224x224 RGB)
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image_array)

@pytest.fixture
def sample_hef_path(test_data_dir):
    """Provide path to a sample HEF file (for testing purposes)."""
    hef_path = test_data_dir / "sample_model.hef"
    
    # Create a mock HEF file if it doesn't exist
    if not hef_path.exists():
        hef_path.parent.mkdir(parents=True, exist_ok=True)
        hef_path.write_bytes(b"MOCK_HEF_DATA")
    
    return hef_path

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    
    # Clean up any test artifacts
    import gc
    gc.collect()

# Custom test markers
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "device: Tests requiring HAILO device",
    "model: Tests requiring model files",
    "slow: Slow running tests",
    "performance: Performance/benchmark tests",
]
