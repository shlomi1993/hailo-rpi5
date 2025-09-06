#!/usr/bin/env python3
"""
Object Detection Example

This example demonstrates object detection using HAILO AI HAT with a YOLO-based detection model.
"""

import sys
import logging
import argparse

from pathlib import Path

from src import HailoDeviceManager, HailoInferenceEngine
from src.utils import PreprocessingUtils, PostprocessingUtils

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


COCO_COMMON_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]


def main():
    parser = argparse.ArgumentParser(description='HAILO Object Detection Example')
    parser.add_argument('--hef', required=True, help='Path to HEF model file')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--device-id', help='Specific device ID to use')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')
    parser.add_argument('--output-format', default='yolo', choices=['yolo', 'ssd'], help='Detection output format')

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

            # Determine target size from input shape (common YOLO sizes)
            if len(input_shape) >= 3:
                if len(input_shape) == 3:
                    if input_shape[2] <= 4:  # Likely (H, W, C)
                        target_size = (input_shape[1], input_shape[0])  # (W, H)
                    else:  # Likely (C, H, W)
                        target_size = (input_shape[2], input_shape[1])  # (W, H)
                else:
                    # Use common YOLO size
                    target_size = (640, 640)
            else:
                target_size = (640, 640)

            logger.info(f"Using target size: {target_size}")

            # Preprocess image
            logger.info(f"Preprocessing image: {args.image}")

            # Use YOLO preprocessing configuration
            preprocessing_config = PreprocessingUtils.get_common_preprocessing_configs()['yolo']
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
            logger.info("Processing detection outputs...")

            # Load class names
            class_names = load_coco_classes()

            # Process detection outputs
            detections = PostprocessingUtils.process_detection_output(
                outputs,
                output_format=args.output_format,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou,
                class_names=class_names
            )

            logger.info(f"Found {len(detections)} detections")

            # Display results
            if detections:
                logger.info("Detection Results:")
                logger.info("-" * 50)
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    logger.info(f"  Detection {i+1}:")
                    logger.info(f"    Class: {detection.get('class_name', 'Unknown')} "
                               f"(ID: {detection['class_id']})")
                    logger.info(f"    Confidence: {detection['confidence']:.3f}")
                    logger.info(f"    Bbox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, "
                               f"{bbox['x2']:.1f}, {bbox['y2']:.1f})")
                    logger.info(f"    Size: {bbox['width']:.1f} x {bbox['height']:.1f}")
                    logger.info("")

                # Create visualization
                visualization = PostprocessingUtils.visualize_detection_results(
                    detections, target_size
                )
                logger.info("\nDetection Summary:")
                logger.info(visualization)

            else:
                logger.info("No detections found above the confidence threshold")

            # Filter specific classes if needed (example: only people and cars)
            person_car_detections = PostprocessingUtils.filter_detections_by_class(
                detections, ['person', 'car']
            )

            if person_car_detections:
                logger.info(f"\nFound {len(person_car_detections)} person/car detections")

            # Run benchmark
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
