#!/usr/bin/env python3
"""
Performance Benchmark Tool

Utility script to benchmark inference performance on HAILO AI HAT
with different models and configurations.
"""

import sys
import logging
import argparse
import time
import statistics
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import HailoDeviceManager, HailoInferenceEngine
from src.utils import PreprocessingUtils
import numpy as np


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_dummy_input(input_specs: dict) -> dict:
    """Create dummy input data for benchmarking."""
    inputs = {}

    for input_name, spec in input_specs.items():
        shape = spec['shape']
        # Create random input data
        if len(shape) == 3:  # Add batch dimension
            shape = [1] + list(shape)
        elif len(shape) == 4 and shape[0] != 1:
            # Adjust batch size to 1
            shape[0] = 1

        # Create random data in [0, 1] range (typical for normalized images)
        dummy_data = np.random.rand(*shape).astype(np.float32)
        inputs[input_name] = dummy_data

    return inputs


def run_warmup(inference_engine, inputs: dict, warmup_iterations: int = 5):
    """Run warmup iterations to stabilize performance."""
    logging.info(f"Running {warmup_iterations} warmup iterations...")

    for i in range(warmup_iterations):
        inference_engine.run_inference(inputs)


def run_latency_benchmark(inference_engine, inputs: dict, iterations: int = 100) -> dict:
    """Run latency benchmark."""
    logging.info(f"Running latency benchmark with {iterations} iterations...")

    latencies = []
    successful_inferences = 0

    for i in range(iterations):
        start_time = time.perf_counter()
        outputs = inference_engine.run_inference(inputs)
        end_time = time.perf_counter()

        if outputs is not None:
            successful_inferences += 1
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            logging.info(f"Completed {i + 1}/{iterations} iterations")

    if latencies:
        results = {
            'iterations': iterations,
            'successful_iterations': successful_inferences,
            'success_rate': successful_inferences / iterations,
            'mean_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'fps': 1000.0 / statistics.mean(latencies)
        }
    else:
        results = {
            'iterations': iterations,
            'successful_iterations': 0,
            'success_rate': 0.0,
            'error': 'No successful inferences'
        }

    return results


def run_throughput_benchmark(inference_engine, inputs: dict, duration_seconds: int = 30) -> dict:
    """Run throughput benchmark for a specified duration."""
    logging.info(f"Running throughput benchmark for {duration_seconds} seconds...")

    start_time = time.time()
    end_time = start_time + duration_seconds

    total_inferences = 0
    successful_inferences = 0

    while time.time() < end_time:
        outputs = inference_engine.run_inference(inputs)
        total_inferences += 1

        if outputs is not None:
            successful_inferences += 1

        if total_inferences % 10 == 0:
            elapsed = time.time() - start_time
            logging.info(f"Progress: {elapsed:.1f}s elapsed, {total_inferences} inferences")

    actual_duration = time.time() - start_time

    results = {
        'duration_seconds': actual_duration,
        'total_inferences': total_inferences,
        'successful_inferences': successful_inferences,
        'success_rate': successful_inferences / total_inferences if total_inferences > 0 else 0,
        'throughput_fps': successful_inferences / actual_duration if actual_duration > 0 else 0,
        'total_throughput_fps': total_inferences / actual_duration if actual_duration > 0 else 0
    }

    return results


def print_benchmark_results(results: dict, benchmark_type: str):
    """Print benchmark results in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"{benchmark_type.upper()} BENCHMARK RESULTS")
    print(f"{'=' * 60}")

    if 'error' in results:
        print(f"❌ Benchmark failed: {results['error']}")
        return

    if benchmark_type == 'latency':
        print(f"Total Iterations: {results['iterations']}")
        print(f"Successful Iterations: {results['successful_iterations']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"\nLatency Statistics:")
        print(f"  Mean: {results['mean_latency_ms']:.2f} ms")
        print(f"  Median: {results['median_latency_ms']:.2f} ms")
        print(f"  Min: {results['min_latency_ms']:.2f} ms")
        print(f"  Max: {results['max_latency_ms']:.2f} ms")
        print(f"  Std Dev: {results['std_latency_ms']:.2f} ms")
        print(f"  95th Percentile: {results['p95_latency_ms']:.2f} ms")
        print(f"  99th Percentile: {results['p99_latency_ms']:.2f} ms")
        print(f"\nThroughput: {results['fps']:.2f} FPS")

    elif benchmark_type == 'throughput':
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Total Inferences: {results['total_inferences']}")
        print(f"Successful Inferences: {results['successful_inferences']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"\nThroughput:")
        print(f"  Successful: {results['throughput_fps']:.2f} FPS")
        print(f"  Total: {results['total_throughput_fps']:.2f} FPS")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='HAILO Performance Benchmark Tool')
    parser.add_argument('--hef', required=True, help='Path to HEF model file')
    parser.add_argument('--device-id', help='Specific device ID to use')
    parser.add_argument('--latency-iterations', type=int, default=100,
                       help='Number of iterations for latency benchmark')
    parser.add_argument('--throughput-duration', type=int, default=30,
                       help='Duration in seconds for throughput benchmark')
    parser.add_argument('--warmup-iterations', type=int, default=5,
                       help='Number of warmup iterations')
    parser.add_argument('--benchmark-type', choices=['latency', 'throughput', 'both'],
                       default='both', help='Type of benchmark to run')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--image', help='Use real image instead of dummy data')

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize device manager
        logger.info("Initializing HAILO device...")
        with HailoDeviceManager() as device_manager:

            # Discover and initialize device
            devices = device_manager.discover_devices()
            if not devices:
                logger.error("No HAILO devices found")
                return 1

            if not device_manager.initialize_device(args.device_id):
                logger.error("Failed to initialize device")
                return 1

            # Load model
            logger.info(f"Loading HEF model: {args.hef}")
            if not device_manager.load_hef(args.hef):
                logger.error("Failed to load HEF model")
                return 1

            # Configure network group
            if not device_manager.configure_network_group():
                logger.error("Failed to configure network group")
                return 1

            # Setup inference engine
            logger.info("Setting up inference engine...")
            inference_engine = HailoInferenceEngine(device_manager)

            if not inference_engine.setup_inference_pipeline():
                logger.error("Failed to setup inference pipeline")
                return 1

            # Get input specifications
            input_specs = inference_engine.get_input_specs()
            logger.info(f"Input specs: {input_specs}")

            # Prepare input data
            if args.image:
                logger.info(f"Using real image: {args.image}")
                # Use the first input spec to determine preprocessing
                first_input_name = next(iter(input_specs.keys()))
                input_shape = input_specs[first_input_name]['shape']

                if len(input_shape) >= 3:
                    if input_shape[2] <= 4:  # Likely (H, W, C)
                        target_size = (input_shape[1], input_shape[0])
                    else:  # Likely (C, H, W)
                        target_size = (input_shape[2], input_shape[1])
                else:
                    target_size = (224, 224)

                # Preprocess the image
                preprocessing_config = PreprocessingUtils.get_common_preprocessing_configs()['simple']
                preprocessing_config['target_size'] = target_size
                preprocessing_fn = PreprocessingUtils.create_preprocessing_pipeline(preprocessing_config)
                preprocessed_image = preprocessing_fn(args.image)

                inputs = {first_input_name: preprocessed_image}
            else:
                logger.info("Using dummy input data")
                inputs = create_dummy_input(input_specs)

            logger.info(f"Input data shapes: {[(name, data.shape) for name, data in inputs.items()]}")

            # Run warmup
            run_warmup(inference_engine, inputs, args.warmup_iterations)

            results = {}

            # Run latency benchmark
            if args.benchmark_type in ['latency', 'both']:
                latency_results = run_latency_benchmark(
                    inference_engine, inputs, args.latency_iterations
                )
                results['latency'] = latency_results
                print_benchmark_results(latency_results, 'latency')

            # Run throughput benchmark
            if args.benchmark_type in ['throughput', 'both']:
                throughput_results = run_throughput_benchmark(
                    inference_engine, inputs, args.throughput_duration
                )
                results['throughput'] = throughput_results
                print_benchmark_results(throughput_results, 'throughput')

            # Add metadata
            results['metadata'] = {
                'hef_file': args.hef,
                'device_info': device_manager.get_device_info(),
                'input_specs': input_specs,
                'benchmark_config': {
                    'latency_iterations': args.latency_iterations,
                    'throughput_duration': args.throughput_duration,
                    'warmup_iterations': args.warmup_iterations,
                    'used_real_image': bool(args.image)
                }
            }

            # Save results if requested
            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {output_path}")

            print(f"\n{'=' * 60}")
            print("BENCHMARK COMPLETE")
            print(f"{'=' * 60}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
