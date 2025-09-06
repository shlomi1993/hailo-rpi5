"""
Unit tests for HailoDeviceManager.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.device_manager import HailoDeviceManager


class TestHailoDeviceManager:
    """Test cases for HailoDeviceManager."""

    @patch('src.core.device_manager.HAILO_AVAILABLE', True)
    @patch('src.core.device_manager.Device')
    def test_discover_devices(self, mock_device_class):
        """Test device discovery functionality."""
        # Setup mock
        mock_device = Mock()
        mock_device.device_id = 'test_device_123'
        mock_device_class.scan.return_value = [mock_device]

        # Test
        manager = HailoDeviceManager()
        devices = manager.discover_devices()

        # Assertions
        assert len(devices) == 1
        assert devices[0] == 'test_device_123'
        mock_device_class.scan.assert_called_once()

    @patch('src.core.device_manager.HAILO_AVAILABLE', False)
    def test_hailo_not_available(self):
        """Test behavior when HAILO is not available."""
        with pytest.raises(RuntimeError, match="PyHailoRT is not available"):
            HailoDeviceManager()

    @patch('src.core.device_manager.HAILO_AVAILABLE', True)
    @patch('src.core.device_manager.Device')
    def test_get_device_by_id(self, mock_device_class):
        """Test getting device by specific ID."""
        # Setup mock
        mock_device = Mock()
        mock_device.device_id = 'target_device'
        mock_device_class.scan.return_value = [mock_device]

        # Test
        manager = HailoDeviceManager()
        device = manager.get_device('target_device')

        # Assertions
        assert device is not None
        assert device.device_id == 'target_device'

    @patch('src.core.device_manager.HAILO_AVAILABLE', True)
    @patch('src.core.device_manager.Device')
    def test_get_device_not_found(self, mock_device_class):
        """Test getting device that doesn't exist."""
        # Setup mock
        mock_device_class.scan.return_value = []

        # Test
        manager = HailoDeviceManager()
        device = manager.get_device('nonexistent_device')

        # Assertions
        assert device is None

    @patch('src.core.device_manager.HAILO_AVAILABLE', True)
    def test_context_manager(self):
        """Test context manager functionality."""
        with patch('src.core.device_manager.Device') as mock_device_class:
            manager = HailoDeviceManager()
            manager.cleanup = Mock()

            with manager:
                pass

            # Should call cleanup when exiting context
            manager.cleanup.assert_called_once()

    @patch('src.core.device_manager.HAILO_AVAILABLE', True)
    @patch('src.core.device_manager.Device')
    def test_device_info_retrieval(self, mock_device_class):
        """Test retrieving device information."""
        # Setup mock
        mock_device = Mock()
        mock_device.device_id = 'test_device'
        mock_device.device_info = Mock()
        mock_device.device_info.device_architecture = 'hailo8'
        mock_device.device_info.device_type = 'AI_HAT'
        mock_device_class.scan.return_value = [mock_device]

        # Test
        manager = HailoDeviceManager()
        device = manager.get_device('test_device')
        
        # Assertions
        assert device.device_info.device_architecture == 'hailo8'
        assert device.device_info.device_type == 'AI_HAT'

    @patch('src.core.device_manager.HAILO_AVAILABLE', True)
    @patch('src.core.device_manager.Device')
    def test_multiple_devices(self, mock_device_class):
        """Test handling multiple devices."""
        # Setup mock
        mock_device1 = Mock()
        mock_device1.device_id = 'device_1'
        mock_device2 = Mock()
        mock_device2.device_id = 'device_2'
        mock_device_class.scan.return_value = [mock_device1, mock_device2]

        # Test
        manager = HailoDeviceManager()
        devices = manager.discover_devices()

        # Assertions
        assert len(devices) == 2
        assert 'device_1' in devices
        assert 'device_2' in devices
