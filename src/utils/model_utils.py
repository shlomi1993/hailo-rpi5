"""
Model Utilities

Utilities for working with HAILO models, including HEF file management,
model information extraction, and model conversion helpers.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json

try:
    import hailo_platform.pyhailort.pyhailort as hailort
    from hailo_platform.pyhailort import HEF
    HAILO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyHailoRT not available: {e}")
    HAILO_AVAILABLE = False


class ModelUtils:
    """Utilities for working with HAILO models and HEF files."""

    @staticmethod
    def analyze_hef(hef_path: str) -> Dict[str, Any]:
        """
        Analyze a HEF file and extract detailed information.

        Args:
            hef_path: Path to the HEF file

        Returns:
            Dictionary containing model information
        """
        if not HAILO_AVAILABLE:
            raise RuntimeError("PyHailoRT is not available")

        try:
            hef_file = Path(hef_path)
            if not hef_file.exists():
                raise FileNotFoundError(f"HEF file not found: {hef_path}")

            hef = HEF(str(hef_file))

            analysis = {
                'file_path': str(hef_file.absolute()),
                'file_size_mb': hef_file.stat().st_size / (1024 * 1024),
                'network_groups': [],
                'total_networks': 0,
                'input_streams': [],
                'output_streams': []
            }

            # Analyze network groups
            network_groups = hef.get_network_groups()
            for ng in network_groups:
                ng_info = {
                    'name': ng.name,
                    'networks': [],
                    'input_streams': [],
                    'output_streams': []
                }

                # Get network information within the group
                networks = ng.get_networks()
                for network in networks:
                    network_info = {
                        'name': network.name,
                        'batch_size': getattr(network, 'batch_size', 'N/A')
                    }
                    ng_info['networks'].append(network_info)

                # Get stream information
                input_streams = ng.get_input_stream_infos()
                for stream in input_streams:
                    stream_info = {
                        'name': stream.name,
                        'shape': stream.shape,
                        'format': str(stream.format),
                        'direction': 'input'
                    }
                    ng_info['input_streams'].append(stream_info)
                    analysis['input_streams'].append(stream_info)

                output_streams = ng.get_output_stream_infos()
                for stream in output_streams:
                    stream_info = {
                        'name': stream.name,
                        'shape': stream.shape,
                        'format': str(stream.format),
                        'direction': 'output'
                    }
                    ng_info['output_streams'].append(stream_info)
                    analysis['output_streams'].append(stream_info)

                analysis['network_groups'].append(ng_info)
                analysis['total_networks'] += len(ng_info['networks'])

            return analysis

        except Exception as e:
            logging.error(f"Error analyzing HEF file: {e}")
            raise

    @staticmethod
    def save_model_info(hef_path: str, output_path: Optional[str] = None) -> str:
        """
        Save model analysis to JSON file.

        Args:
            hef_path: Path to the HEF file
            output_path: Output path for JSON file (optional)

        Returns:
            Path to the saved JSON file
        """
        analysis = ModelUtils.analyze_hef(hef_path)

        if output_path is None:
            hef_file = Path(hef_path)
            output_path = hef_file.parent / f"{hef_file.stem}_info.json"

        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        logging.info(f"Model analysis saved to: {output_file}")
        return str(output_file)

    @staticmethod
    def get_model_requirements(hef_path: str) -> Dict[str, Any]:
        """
        Get model requirements for deployment.

        Args:
            hef_path: Path to the HEF file

        Returns:
            Dictionary containing deployment requirements
        """
        analysis = ModelUtils.analyze_hef(hef_path)

        requirements = {
            'min_memory_mb': analysis['file_size_mb'] * 2,  # Rough estimate
            'input_formats': [],
            'output_formats': [],
            'batch_sizes': [],
            'stream_interface': 'PCIe'  # Default for AI HAT
        }

        # Extract unique formats and batch sizes
        input_formats = set()
        output_formats = set()

        for ng in analysis['network_groups']:
            for stream in ng['input_streams']:
                input_formats.add(stream['format'])
            for stream in ng['output_streams']:
                output_formats.add(stream['format'])

            for network in ng['networks']:
                if network['batch_size'] != 'N/A':
                    requirements['batch_sizes'].append(network['batch_size'])

        requirements['input_formats'] = list(input_formats)
        requirements['output_formats'] = list(output_formats)
        requirements['batch_sizes'] = list(set(requirements['batch_sizes']))

        return requirements

    @staticmethod
    def validate_hef_compatibility(hef_path: str) -> Tuple[bool, List[str]]:
        """
        Validate HEF file compatibility with current platform.

        Args:
            hef_path: Path to the HEF file

        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []

        try:
            if not HAILO_AVAILABLE:
                issues.append("PyHailoRT is not available")
                return False, issues

            # Check file existence
            hef_file = Path(hef_path)
            if not hef_file.exists():
                issues.append(f"HEF file not found: {hef_path}")
                return False, issues

            # Try to load the HEF
            try:
                hef = HEF(str(hef_file))
                network_groups = hef.get_network_groups()

                if not network_groups:
                    issues.append("No network groups found in HEF")

                # Check for common compatibility issues
                for ng in network_groups:
                    input_streams = ng.get_input_stream_infos()
                    output_streams = ng.get_output_stream_infos()

                    if not input_streams:
                        issues.append(f"Network group '{ng.name}' has no input streams")

                    if not output_streams:
                        issues.append(f"Network group '{ng.name}' has no output streams")

            except Exception as e:
                issues.append(f"Failed to load HEF: {str(e)}")

        except Exception as e:
            issues.append(f"Validation error: {str(e)}")

        is_compatible = len(issues) == 0
        return is_compatible, issues

    @staticmethod
    def list_model_files(directory: str, extensions: List[str] = None) -> List[str]:
        """
        List model files in a directory.

        Args:
            directory: Directory to search
            extensions: List of file extensions to include (default: ['.hef'])

        Returns:
            List of model file paths
        """
        if extensions is None:
            extensions = ['.hef']

        directory = Path(directory)
        if not directory.exists():
            logging.warning(f"Directory not found: {directory}")
            return []

        model_files = []
        for ext in extensions:
            model_files.extend(directory.glob(f"*{ext}"))
            model_files.extend(directory.glob(f"**/*{ext}"))  # Recursive search

        # Remove duplicates and convert to strings
        unique_files = list(set(str(f) for f in model_files))
        unique_files.sort()

        logging.info(f"Found {len(unique_files)} model files in {directory}")
        return unique_files

    @staticmethod
    def estimate_inference_memory(analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate memory requirements for inference.

        Args:
            analysis: Model analysis from analyze_hef()

        Returns:
            Dictionary with memory estimates in MB
        """
        estimates = {
            'model_size_mb': analysis['file_size_mb'],
            'input_buffers_mb': 0.0,
            'output_buffers_mb': 0.0,
            'total_estimated_mb': 0.0
        }

        # Estimate buffer sizes (very rough calculation)
        for stream in analysis['input_streams']:
            if 'shape' in stream and stream['shape']:
                # Assume float32 (4 bytes per element)
                elements = 1
                for dim in stream['shape']:
                    elements *= dim
                estimates['input_buffers_mb'] += (elements * 4) / (1024 * 1024)

        for stream in analysis['output_streams']:
            if 'shape' in stream and stream['shape']:
                # Assume float32 (4 bytes per element)
                elements = 1
                for dim in stream['shape']:
                    elements *= dim
                estimates['output_buffers_mb'] += (elements * 4) / (1024 * 1024)

        estimates['total_estimated_mb'] = (
            estimates['model_size_mb'] +
            estimates['input_buffers_mb'] +
            estimates['output_buffers_mb'] +
            50  # Additional overhead estimate
        )

        return estimates
