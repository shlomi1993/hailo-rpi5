#!/usr/bin/env python3
"""
Model Analysis Tool

Utility script to analyze HEF model files and extract detailed information
about their structure, inputs, outputs, and requirements.
"""

import sys
import logging
import argparse
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ModelUtils


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_analysis(analysis: dict):
    """Print model analysis in a readable format."""
    print("=" * 60)
    print("HAILO MODEL ANALYSIS")
    print("=" * 60)

    print(f"\nFile Information:")
    print(f"  Path: {analysis['file_path']}")
    print(f"  Size: {analysis['file_size_mb']:.2f} MB")
    print(f"  Total Networks: {analysis['total_networks']}")
    print(f"  Network Groups: {len(analysis['network_groups'])}")

    print(f"\nNetwork Groups:")
    for i, ng in enumerate(analysis['network_groups']):
        print(f"  {i+1}. {ng['name']}")
        print(f"     Networks: {len(ng['networks'])}")
        print(f"     Input Streams: {len(ng['input_streams'])}")
        print(f"     Output Streams: {len(ng['output_streams'])}")

        if ng['networks']:
            print(f"     Network Details:")
            for net in ng['networks']:
                print(f"       - {net['name']} (batch_size: {net['batch_size']})")

    print(f"\nInput Streams:")
    for i, stream in enumerate(analysis['input_streams']):
        print(f"  {i+1}. {stream['name']}")
        print(f"     Shape: {stream['shape']}")
        print(f"     Format: {stream['format']}")

    print(f"\nOutput Streams:")
    for i, stream in enumerate(analysis['output_streams']):
        print(f"  {i+1}. {stream['name']}")
        print(f"     Shape: {stream['shape']}")
        print(f"     Format: {stream['format']}")


def print_requirements(requirements: dict):
    """Print model requirements."""
    print("\n" + "=" * 60)
    print("DEPLOYMENT REQUIREMENTS")
    print("=" * 60)

    print(f"\nMemory Requirements:")
    print(f"  Minimum Memory: {requirements['min_memory_mb']:.2f} MB")
    print(f"  Stream Interface: {requirements['stream_interface']}")

    print(f"\nInput Formats: {', '.join(requirements['input_formats'])}")
    print(f"Output Formats: {', '.join(requirements['output_formats'])}")

    if requirements['batch_sizes']:
        print(f"Batch Sizes: {', '.join(map(str, requirements['batch_sizes']))}")


def print_memory_estimates(estimates: dict):
    """Print memory estimates."""
    print("\n" + "=" * 60)
    print("MEMORY ESTIMATES")
    print("=" * 60)

    print(f"\nModel Size: {estimates['model_size_mb']:.2f} MB")
    print(f"Input Buffers: {estimates['input_buffers_mb']:.2f} MB")
    print(f"Output Buffers: {estimates['output_buffers_mb']:.2f} MB")
    print(f"Total Estimated: {estimates['total_estimated_mb']:.2f} MB")


def validate_and_report(hef_path: str):
    """Validate HEF file and report issues."""
    print("\n" + "=" * 60)
    print("COMPATIBILITY VALIDATION")
    print("=" * 60)

    is_compatible, issues = ModelUtils.validate_hef_compatibility(hef_path)

    if is_compatible:
        print("\n✅ Model is compatible with the current platform")
    else:
        print("\n❌ Model compatibility issues found:")
        for issue in issues:
            print(f"   - {issue}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='HAILO Model Analysis Tool')
    parser.add_argument('hef_file', help='Path to HEF model file')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--validate', action='store_true',
                       help='Validate model compatibility')
    parser.add_argument('--list-models', '-l',
                       help='List all HEF files in directory')

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # List models if requested
        if args.list_models:
            models = ModelUtils.list_model_files(args.list_models)
            print(f"\nFound {len(models)} HEF files:")
            for model in models:
                print(f"  - {model}")
            return 0

        # Check if HEF file exists
        hef_path = Path(args.hef_file)
        if not hef_path.exists():
            logger.error(f"HEF file not found: {args.hef_file}")
            return 1

        logger.info(f"Analyzing HEF file: {args.hef_file}")

        # Analyze the model
        analysis = ModelUtils.analyze_hef(str(hef_path))

        # Print analysis
        print_analysis(analysis)

        # Get and print requirements
        requirements = ModelUtils.get_model_requirements(str(hef_path))
        print_requirements(requirements)

        # Estimate memory usage
        memory_estimates = ModelUtils.estimate_inference_memory(analysis)
        print_memory_estimates(memory_estimates)

        # Validate compatibility if requested
        if args.validate:
            validate_and_report(str(hef_path))

        # Save to JSON if requested
        if args.output:
            output_data = {
                'analysis': analysis,
                'requirements': requirements,
                'memory_estimates': memory_estimates
            }

            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Analysis saved to: {output_path}")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
