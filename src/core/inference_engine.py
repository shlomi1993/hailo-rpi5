"""
HAILO Inference Engine

Provides high-level interface for running inference on HAILO AI HAT
using PyHailoRT with support for various input/output formats.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import numpy as np
from pathlib import Path
import time

try:
    import hailo_platform.pyhailort.pyhailort as hailort
    from hailo_platform.pyhailort import InferVStreams, InputVStreamParams, OutputVStreamParams
    from hailo_platform.pyhailort import HailoStreamInterface, FormatType
    HAILO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyHailoRT not available: {e}")
    HAILO_AVAILABLE = False

from .device_manager import HailoDeviceManager


class HailoInferenceEngine:
    """
    High-level inference engine for HAILO AI HAT.

    Provides easy-to-use interface for running inference with preprocessing
    and postprocessing capabilities.
    """

    def __init__(self, device_manager: HailoDeviceManager):
        """
        Initialize inference engine.

        Args:
            device_manager: Initialized HailoDeviceManager instance
        """
        self.device_manager = device_manager
        self.infer_pipeline: Optional[InferVStreams] = None
        self.input_vstreams_params = {}
        self.output_vstreams_params = {}
        self.logger = logging.getLogger(__name__)

        if not HAILO_AVAILABLE:
            raise RuntimeError("PyHailoRT is not available.")

    def setup_inference_pipeline(self, network_group_name: Optional[str] = None,
                                batch_size: int = 1) -> bool:
        """
        Setup inference pipeline for the specified network group.

        Args:
            network_group_name: Name of network group to use
            batch_size: Batch size for inference

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the configured network group
            if network_group_name:
                if network_group_name not in self.device_manager.network_groups:
                    self.logger.error(f"Network group '{network_group_name}' not configured")
                    return False
                network_group = self.device_manager.network_groups[network_group_name]
            else:
                # Use first available network group
                if not self.device_manager.network_groups:
                    self.logger.error("No network groups configured")
                    return False
                network_group = next(iter(self.device_manager.network_groups.values()))

            # Get input and output stream information
            input_vstream_infos = network_group.get_input_vstream_infos()
            output_vstream_infos = network_group.get_output_vstream_infos()

            # Setup input parameters
            for input_info in input_vstream_infos:
                self.input_vstreams_params[input_info.name] = InputVStreamParams.make_from_network_group(
                    network_group, quantized=False, format_type=FormatType.FLOAT32
                )[input_info.name]

                self.logger.info(f"Input stream '{input_info.name}': "
                               f"shape={input_info.shape}, format={input_info.format}")

            # Setup output parameters
            for output_info in output_vstream_infos:
                self.output_vstreams_params[output_info.name] = OutputVStreamParams.make_from_network_group(
                    network_group, quantized=False, format_type=FormatType.FLOAT32
                )[output_info.name]

                self.logger.info(f"Output stream '{output_info.name}': "
                               f"shape={output_info.shape}, format={output_info.format}")

            # Create inference pipeline
            self.infer_pipeline = InferVStreams(network_group,
                                              self.input_vstreams_params,
                                              self.output_vstreams_params)

            self.logger.info("Inference pipeline setup complete")
            return True

        except Exception as e:
            self.logger.error(f"Error setting up inference pipeline: {e}")
            return False

    def run_inference(self, inputs: Dict[str, np.ndarray],
                     timeout_ms: int = 10000) -> Optional[Dict[str, np.ndarray]]:
        """
        Run inference with the given inputs.

        Args:
            inputs: Dictionary mapping input stream names to numpy arrays
            timeout_ms: Timeout in milliseconds

        Returns:
            Dictionary mapping output stream names to numpy arrays, or None if failed
        """
        try:
            if not self.infer_pipeline:
                self.logger.error("Inference pipeline not setup")
                return None

            # Validate inputs
            for stream_name, data in inputs.items():
                if stream_name not in self.input_vstreams_params:
                    self.logger.error(f"Unknown input stream: {stream_name}")
                    return None

                if not isinstance(data, np.ndarray):
                    self.logger.error(f"Input data for '{stream_name}' must be numpy array")
                    return None

            # Run inference
            start_time = time.time()
            with self.infer_pipeline as infer:
                outputs = infer.infer(inputs, timeout=timeout_ms)

            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.logger.info(f"Inference completed in {inference_time:.2f}ms")

            return outputs

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return None

    def run_inference_async(self, inputs: Dict[str, np.ndarray],
                           callback=None) -> bool:
        """
        Run asynchronous inference (placeholder for future implementation).

        Args:
            inputs: Dictionary mapping input stream names to numpy arrays
            callback: Optional callback function for results

        Returns:
            True if inference started successfully
        """
        # TODO: Implement async inference using HailoRT async capabilities
        self.logger.warning("Async inference not yet implemented")
        return False

    def get_input_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get specifications for input streams.

        Returns:
            Dictionary mapping input names to their specifications
        """
        specs = {}

        if not self.device_manager.network_groups:
            return specs

        try:
            # Get first network group for now
            network_group = next(iter(self.device_manager.network_groups.values()))
            input_vstream_infos = network_group.get_input_vstream_infos()

            for input_info in input_vstream_infos:
                specs[input_info.name] = {
                    'shape': input_info.shape,
                    'format': str(input_info.format),
                    'direction': str(input_info.direction),
                    'network_name': input_info.network_name
                }

                # Add quantization info if available
                if hasattr(input_info, 'quant_info'):
                    specs[input_info.name]['quant_info'] = {
                        'qp_zp': input_info.quant_info.qp_zp,
                        'qp_scale': input_info.quant_info.qp_scale
                    }

        except Exception as e:
            self.logger.error(f"Error getting input specs: {e}")

        return specs

    def get_output_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get specifications for output streams.

        Returns:
            Dictionary mapping output names to their specifications
        """
        specs = {}

        if not self.device_manager.network_groups:
            return specs

        try:
            # Get first network group for now
            network_group = next(iter(self.device_manager.network_groups.values()))
            output_vstream_infos = network_group.get_output_vstream_infos()

            for output_info in output_vstream_infos:
                specs[output_info.name] = {
                    'shape': output_info.shape,
                    'format': str(output_info.format),
                    'direction': str(output_info.direction),
                    'network_name': output_info.network_name
                }

                # Add quantization info if available
                if hasattr(output_info, 'quant_info'):
                    specs[output_info.name]['quant_info'] = {
                        'qp_zp': output_info.quant_info.qp_zp,
                        'qp_scale': output_info.quant_info.qp_scale
                    }

        except Exception as e:
            self.logger.error(f"Error getting output specs: {e}")

        return specs

    def benchmark_inference(self, inputs: Dict[str, np.ndarray],
                           num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            inputs: Dictionary mapping input stream names to numpy arrays
            num_iterations: Number of iterations to run

        Returns:
            Dictionary with performance metrics
        """
        if not self.infer_pipeline:
            self.logger.error("Inference pipeline not setup")
            return {}

        inference_times = []
        successful_inferences = 0

        self.logger.info(f"Starting benchmark with {num_iterations} iterations...")

        for i in range(num_iterations):
            start_time = time.time()
            outputs = self.run_inference(inputs)
            end_time = time.time()

            if outputs is not None:
                successful_inferences += 1
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms

            if (i + 1) % 10 == 0:
                self.logger.info(f"Completed {i + 1}/{num_iterations} iterations")

        if inference_times:
            metrics = {
                'total_iterations': num_iterations,
                'successful_iterations': successful_inferences,
                'success_rate': successful_inferences / num_iterations,
                'avg_inference_time_ms': np.mean(inference_times),
                'min_inference_time_ms': np.min(inference_times),
                'max_inference_time_ms': np.max(inference_times),
                'std_inference_time_ms': np.std(inference_times),
                'fps': 1000.0 / np.mean(inference_times) if inference_times else 0.0
            }
        else:
            metrics = {
                'total_iterations': num_iterations,
                'successful_iterations': 0,
                'success_rate': 0.0,
                'error': 'No successful inferences'
            }

        self.logger.info(f"Benchmark completed: {metrics}")
        return metrics

    def cleanup(self):
        """Clean up inference pipeline resources."""
        try:
            if self.infer_pipeline:
                # The pipeline will be cleaned up when exiting context
                self.infer_pipeline = None
                self.logger.info("Cleaned up inference pipeline")
        except Exception as e:
            self.logger.error(f"Error during inference engine cleanup: {e}")
