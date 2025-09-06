"""
HAILO AI HAT Application for Raspberry Pi 5

This package provides utilities and examples for developing AI applications
using the HAILO AI HAT on Raspberry Pi 5 with LibHailoRT and PyHailoRT.
"""

__version__ = "0.1.0"
__author__ = "HAILO RPI5 Developer"

# Core imports for easy access
from .core.device_manager import HailoDeviceManager
from .core.inference_engine import HailoInferenceEngine
from .utils.model_utils import ModelUtils
from .utils.preprocessing import PreprocessingUtils

__all__ = [
    "HailoDeviceManager",
    "HailoInferenceEngine",
    "ModelUtils",
    "PreprocessingUtils"
]
