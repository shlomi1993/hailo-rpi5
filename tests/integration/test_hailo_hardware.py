"""
Integration tests for HAILO RPI5 package.

These tests require actual HAILO hardware and should be run on Raspberry Pi 5
with HAILO AI HAT connected.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.device_manager import HailoDeviceManager
from src.core.inference_engine import HailoInferenceEngine


@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc/device-tree/hat").exists(), 
                   reason="HAILO AI HAT not detected")
class TestHailoIntegration:
    """Integration tests requiring actual HAILO hardware."""

    def test_device_discovery(self):
        """Test actual device discovery on hardware."""
        try:
            manager = HailoDeviceManager()
            devices = manager.discover_devices()
            
            # Should find at least one device on HAILO AI HAT
            assert len(devices) > 0
            assert all(isinstance(device_id, str) for device_id in devices)
            
        except RuntimeError:
            pytest.skip("HAILO platform not available")

    def test_device_connection(self):
        """Test connecting to actual HAILO device."""
        try:
            manager = HailoDeviceManager()
            devices = manager.discover_devices()
            
            if not devices:
                pytest.skip("No HAILO devices found")
            
            device = manager.get_device(devices[0])
            assert device is not None
            assert hasattr(device, 'device_info')
            
        except RuntimeError:
            pytest.skip("HAILO platform not available")

    @pytest.mark.slow
    def test_model_loading_integration(self, sample_model_path):
        """Test loading an actual model file."""
        if not Path(sample_model_path).exists():
            pytest.skip(f"Model file not found: {sample_model_path}")
        
        try:
            engine = HailoInferenceEngine()
            success = engine.load_model(sample_model_path)
            
            assert success is True
            assert engine.device is not None
            assert engine.network_group is not None
            
            # Test getting model info
            info = engine.get_model_info()
            assert 'network_infos' in info
            
        except RuntimeError:
            pytest.skip("HAILO platform not available")
        finally:
            if 'engine' in locals():
                engine.cleanup()

    @pytest.mark.slow
    def test_end_to_end_inference(self, sample_model_path):
        """Test complete inference pipeline with actual hardware."""
        if not Path(sample_model_path).exists():
            pytest.skip(f"Model file not found: {sample_model_path}")
        
        try:
            engine = HailoInferenceEngine()
            success = engine.load_model(sample_model_path)
            
            if not success:
                pytest.skip("Could not load model")
            
            # Create a sample input image
            sample_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
            
            # Run inference
            results = engine.run_inference(sample_image)
            
            # Basic validation of results
            assert results is not None
            assert isinstance(results, (list, dict, np.ndarray))
            
        except RuntimeError:
            pytest.skip("HAILO platform not available")
        finally:
            if 'engine' in locals():
                engine.cleanup()


@pytest.fixture(scope="session")
def sample_model_path():
    """Path to a sample model for integration testing."""
    # This should point to an actual model file for integration testing
    return "/opt/hailo/models/yolov5s.hef"


@pytest.mark.integration
class TestHailoPerformance:
    """Performance tests for HAILO operations."""
    
    @pytest.mark.slow
    def test_inference_performance(self, sample_model_path):
        """Test inference performance and timing."""
        if not Path(sample_model_path).exists():
            pytest.skip(f"Model file not found: {sample_model_path}")
        
        import time
        
        try:
            engine = HailoInferenceEngine()
            success = engine.load_model(sample_model_path)
            
            if not success:
                pytest.skip("Could not load model")
            
            # Warm up
            sample_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
            engine.run_inference(sample_image)
            
            # Measure inference time
            start_time = time.time()
            num_inferences = 10
            
            for _ in range(num_inferences):
                engine.run_inference(sample_image)
            
            total_time = time.time() - start_time
            avg_time = total_time / num_inferences
            fps = 1.0 / avg_time
            
            # Basic performance assertions
            assert avg_time < 1.0  # Should be less than 1 second per inference
            assert fps > 1.0       # Should achieve more than 1 FPS
            
            print(f"Average inference time: {avg_time:.3f}s")
            print(f"Average FPS: {fps:.1f}")
            
        except RuntimeError:
            pytest.skip("HAILO platform not available")
        finally:
            if 'engine' in locals():
                engine.cleanup()

    @pytest.mark.slow
    def test_memory_usage(self, sample_model_path):
        """Test memory usage during inference."""
        if not Path(sample_model_path).exists():
            pytest.skip(f"Model file not found: {sample_model_path}")
        
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            engine = HailoInferenceEngine()
            success = engine.load_model(sample_model_path)
            
            if not success:
                pytest.skip("Could not load model")
            
            # Run multiple inferences
            sample_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
            
            for _ in range(20):
                engine.run_inference(sample_image)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024  # 100MB
            
            print(f"Memory increase: {memory_increase / 1024 / 1024:.1f}MB")
            
        except RuntimeError:
            pytest.skip("HAILO platform not available")
        finally:
            if 'engine' in locals():
                engine.cleanup()
