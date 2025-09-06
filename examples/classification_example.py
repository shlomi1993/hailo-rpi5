#!/usr/bin/env python3
"""
Simple Classification Example

This example demonstrates basic image classification using HAILO AI HA with a pre-trained model.
"""

import sys
import logging
import argparse

from pathlib import Path

from src import HailoDeviceManager, HailoInferenceEngine
from src.utils import PreprocessingUtils, PostprocessingUtils


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_imagenet_classes() -> list:
    return [f"class_{i}" for i in range(1000)]  # In a real implementation, you would load this from a file


def main():
    parser = argparse.ArgumentParser(description='HAILO Classification Example')
    parser.add_argument('--hef', required=True, help='Path to HEF model file')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--device-id', help='Specific device ID to use')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

            # Print device info
            device_info = device_manager.get_device_info()
            logger.info(f"Device Info: {device_info}")

            # Setup inference engine
            logger.info("Setting up inference engine...")
            inference_engine = HailoInferenceEngine(device_manager)

            if not inference_engine.setup_inference_pipeline():
                logger.error("Failed to setup inference pipeline")
                return 1

            # Get input specifications
            input_specs = inference_engine.get_input_specs()
            output_specs = inference_engine.get_output_specs()

            logger.info(f"Input specs: {input_specs}")
            logger.info(f"Output specs: {output_specs}")

            # Get the first input stream info for preprocessing
            if not input_specs:
                logger.error("No input streams found")
                return 1

            first_input_name = next(iter(input_specs.keys()))
            input_shape = input_specs[first_input_name]['shape']

            # Determine target size from input shape
            if len(input_shape) >= 3:
                # Assuming format is (H, W, C) or (C, H, W)
                if len(input_shape) == 3:
                    if input_shape[2] <= 4:  # Likely (H, W, C)
                        target_size = (input_shape[1], input_shape[0])  # (W, H)
                    else:  # Likely (C, H, W)
                        target_size = (input_shape[2], input_shape[1])  # (W, H)
                else:
                    # Use default
                    target_size = (224, 224)
            else:
                target_size = (224, 224)

            logger.info(f"Using target size: {target_size}")

            # Preprocess image
            logger.info(f"Preprocessing image: {args.image}")

            # Use simple preprocessing configuration
            preprocessing_config = PreprocessingUtils.get_common_preprocessing_configs()['simple']
            preprocessing_config['target_size'] = target_size

            preprocessing_fn = PreprocessingUtils.create_preprocessing_pipeline(preprocessing_config)
            preprocessed_image = preprocessing_fn(args.image)

            logger.info(f"Preprocessed image shape: {preprocessed_image.shape}")

            # Prepare inputs
            inputs = {first_input_name: preprocessed_image}

            # Run inference
            logger.info("Running inference...")
            outputs = inference_engine.run_inference(inputs)

            if outputs is None:
                logger.error("Inference failed")
                return 1

            logger.info("Inference completed successfully")

            # Process outputs
            logger.info("Processing outputs...")

            # Get the first output for classification
            first_output_name = next(iter(outputs.keys()))
            output_data = outputs[first_output_name]

            logger.info(f"Output shape: {output_data.shape}")

            # Apply softmax if needed (assuming logits)
            probabilities = PostprocessingUtils.softmax(output_data)

            # Get top-k predictions
            class_names = load_imagenet_classes()
            predictions = PostprocessingUtils.get_top_k_predictions(
                probabilities, k=args.top_k, class_names=class_names
            )

            # Display results
            logger.info("Classification Results:")
            logger.info("-" * 40)
            for pred in predictions:
                logger.info(f"  {pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")

            # Run benchmark if requested
            logger.info("Running performance benchmark...")
            benchmark_results = inference_engine.benchmark_inference(inputs, num_iterations=10)
            logger.info(f"Benchmark results: {benchmark_results}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
