"""
Postprocessing Utilities

Common postprocessing functions for interpreting HAILO AI HAT inference outputs,
including classification, detection, and segmentation results.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import numpy as np


class PostprocessingUtils:
    """Utilities for postprocessing HAILO inference outputs."""

    @staticmethod
    def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply softmax activation to logits.

        Args:
            logits: Input logits
            axis: Axis along which to apply softmax

        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        shifted_logits = logits - np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

    @staticmethod
    def get_top_k_predictions(probabilities: np.ndarray, k: int = 5,
                             class_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get top-k predictions from classification probabilities.

        Args:
            probabilities: Classification probabilities
            k: Number of top predictions to return
            class_names: Optional list of class names

        Returns:
            List of dictionaries with top-k predictions
        """
        if len(probabilities.shape) > 1:
            # Remove batch dimension if present
            probabilities = probabilities.flatten()

        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[::-1][:k]

        predictions = []
        for i, idx in enumerate(top_k_indices):
            prediction = {
                'rank': i + 1,
                'class_id': int(idx),
                'probability': float(probabilities[idx]),
                'confidence': float(probabilities[idx])
            }

            if class_names and idx < len(class_names):
                prediction['class_name'] = class_names[idx]
            else:
                prediction['class_name'] = f"class_{idx}"

            predictions.append(prediction)

        return predictions

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, score_threshold: float = 0.5,
            iou_threshold: float = 0.5) -> List[int]:
        """
        Apply Non-Maximum Suppression to detection results.

        Args:
            boxes: Bounding boxes array (N, 4) in format [x1, y1, x2, y2]
            scores: Confidence scores array (N,)
            score_threshold: Minimum score threshold
            iou_threshold: IoU threshold for suppression

        Returns:
            List of indices of boxes to keep
        """
        if len(boxes) == 0:
            return []

        # Filter by score threshold
        valid_indices = scores >= score_threshold
        if not np.any(valid_indices):
            return []

        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        original_indices = np.where(valid_indices)[0]

        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Sort by scores in descending order
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # Pick the box with highest score
            i = order[0]
            keep.append(original_indices[i])

            if len(order) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union

            # Keep boxes with IoU less than threshold
            order = order[1:][iou <= iou_threshold]

        return keep

    @staticmethod
    def process_detection_output(outputs: Dict[str, np.ndarray],
                                output_format: str = 'yolo',
                                confidence_threshold: float = 0.5,
                                iou_threshold: float = 0.5,
                                class_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process detection model outputs.

        Args:
            outputs: Dictionary of model outputs
            output_format: Format of outputs ('yolo', 'ssd', 'rcnn')
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            class_names: Optional list of class names

        Returns:
            List of detection dictionaries
        """
        detections = []

        if output_format.lower() == 'yolo':
            # Assuming YOLO format: [batch, grid_y, grid_x, anchors, (x, y, w, h, conf, class_probs...)]
            for output_name, output_data in outputs.items():
                if len(output_data.shape) >= 3:
                    detections.extend(PostprocessingUtils._process_yolo_output(
                        output_data, confidence_threshold, iou_threshold, class_names
                    ))

        elif output_format.lower() == 'ssd':
            # Handle SSD format outputs
            detections.extend(PostprocessingUtils._process_ssd_output(
                outputs, confidence_threshold, iou_threshold, class_names
            ))

        else:
            logging.warning(f"Unsupported detection format: {output_format}")

        return detections

    @staticmethod
    def _process_yolo_output(output_data: np.ndarray, confidence_threshold: float,
                           iou_threshold: float, class_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Process YOLO format output."""
        detections = []

        # Remove batch dimension if present
        if len(output_data.shape) > 3 and output_data.shape[0] == 1:
            output_data = output_data[0]

        # Flatten spatial dimensions if needed
        if len(output_data.shape) == 4:
            grid_h, grid_w, anchors, features = output_data.shape
            output_data = output_data.reshape(-1, features)

        # Extract boxes, confidence, and class probabilities
        if output_data.shape[1] >= 5:
            boxes = output_data[:, :4]  # x, y, w, h
            confidence = output_data[:, 4]

            if output_data.shape[1] > 5:
                class_probs = output_data[:, 5:]
                scores = confidence.reshape(-1, 1) * class_probs
                class_ids = np.argmax(scores, axis=1)
                max_scores = np.max(scores, axis=1)
            else:
                max_scores = confidence
                class_ids = np.zeros(len(confidence), dtype=int)

            # Filter by confidence
            valid_mask = max_scores >= confidence_threshold
            if np.any(valid_mask):
                boxes = boxes[valid_mask]
                scores = max_scores[valid_mask]
                class_ids = class_ids[valid_mask]

                # Convert center format to corner format for NMS
                x1 = boxes[:, 0] - boxes[:, 2] / 2
                y1 = boxes[:, 1] - boxes[:, 3] / 2
                x2 = boxes[:, 0] + boxes[:, 2] / 2
                y2 = boxes[:, 1] + boxes[:, 3] / 2
                corner_boxes = np.stack([x1, y1, x2, y2], axis=1)

                # Apply NMS
                keep_indices = PostprocessingUtils.nms(corner_boxes, scores,
                                                     confidence_threshold, iou_threshold)

                # Create detection results
                for idx in keep_indices:
                    detection = {
                        'bbox': {
                            'x1': float(x1[idx]),
                            'y1': float(y1[idx]),
                            'x2': float(x2[idx]),
                            'y2': float(y2[idx]),
                            'center_x': float(boxes[idx, 0]),
                            'center_y': float(boxes[idx, 1]),
                            'width': float(boxes[idx, 2]),
                            'height': float(boxes[idx, 3])
                        },
                        'confidence': float(scores[idx]),
                        'class_id': int(class_ids[idx])
                    }

                    if class_names and class_ids[idx] < len(class_names):
                        detection['class_name'] = class_names[class_ids[idx]]
                    else:
                        detection['class_name'] = f"class_{class_ids[idx]}"

                    detections.append(detection)

        return detections

    @staticmethod
    def _process_ssd_output(outputs: Dict[str, np.ndarray], confidence_threshold: float,
                          iou_threshold: float, class_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Process SSD format output."""
        # Placeholder for SSD processing
        # SSD typically has separate outputs for boxes, scores, and classes
        detections = []
        logging.warning("SSD postprocessing not fully implemented")
        return detections

    @staticmethod
    def process_segmentation_output(outputs: Dict[str, np.ndarray],
                                  num_classes: Optional[int] = None,
                                  apply_softmax: bool = True) -> Dict[str, Any]:
        """
        Process segmentation model outputs.

        Args:
            outputs: Dictionary of model outputs
            num_classes: Number of classes (inferred if not provided)
            apply_softmax: Whether to apply softmax to logits

        Returns:
            Dictionary containing segmentation results
        """
        results = {}

        for output_name, output_data in outputs.items():
            # Remove batch dimension if present
            if len(output_data.shape) == 4 and output_data.shape[0] == 1:
                output_data = output_data[0]

            if apply_softmax and len(output_data.shape) == 3:
                # Apply softmax along channel dimension
                probabilities = PostprocessingUtils.softmax(output_data, axis=-1)
            else:
                probabilities = output_data

            # Get class predictions
            if len(probabilities.shape) == 3:
                class_predictions = np.argmax(probabilities, axis=-1)
                max_probabilities = np.max(probabilities, axis=-1)
            else:
                class_predictions = probabilities
                max_probabilities = None

            results[output_name] = {
                'class_predictions': class_predictions,
                'probabilities': probabilities,
                'max_probabilities': max_probabilities,
                'shape': output_data.shape
            }

        return results

    @staticmethod
    def visualize_detection_results(detections: List[Dict[str, Any]],
                                  image_shape: Tuple[int, int]) -> str:
        """
        Create a text visualization of detection results.

        Args:
            detections: List of detection dictionaries
            image_shape: Shape of the original image (height, width)

        Returns:
            String representation of detections
        """
        if not detections:
            return "No detections found."

        visualization = f"Found {len(detections)} detections:\n"
        visualization += "-" * 50 + "\n"

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            visualization += f"Detection {i+1}:\n"
            visualization += f"  Class: {detection.get('class_name', 'Unknown')} (ID: {detection['class_id']})\n"
            visualization += f"  Confidence: {detection['confidence']:.3f}\n"
            visualization += f"  Bbox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f})\n"
            visualization += f"  Size: {bbox['width']:.1f} x {bbox['height']:.1f}\n"
            visualization += "\n"

        return visualization

    @staticmethod
    def filter_detections_by_class(detections: List[Dict[str, Any]],
                                 target_classes: Union[List[str], List[int]]) -> List[Dict[str, Any]]:
        """
        Filter detections by class names or IDs.

        Args:
            detections: List of detection dictionaries
            target_classes: List of class names or IDs to keep

        Returns:
            Filtered list of detections
        """
        filtered = []

        for detection in detections:
            if isinstance(target_classes[0], str):
                # Filter by class names
                if detection.get('class_name') in target_classes:
                    filtered.append(detection)
            else:
                # Filter by class IDs
                if detection['class_id'] in target_classes:
                    filtered.append(detection)

        return filtered

    @staticmethod
    def scale_detections_to_image(detections: List[Dict[str, Any]],
                                model_input_size: Tuple[int, int],
                                image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Scale detection coordinates from model input size to original image size.

        Args:
            detections: List of detection dictionaries
            model_input_size: Size of model input (width, height)
            image_size: Size of original image (width, height)

        Returns:
            List of detections with scaled coordinates
        """
        scale_x = image_size[0] / model_input_size[0]
        scale_y = image_size[1] / model_input_size[1]

        scaled_detections = []
        for detection in detections:
            scaled_detection = detection.copy()
            bbox = detection['bbox'].copy()

            # Scale coordinates
            bbox['x1'] *= scale_x
            bbox['x2'] *= scale_x
            bbox['y1'] *= scale_y
            bbox['y2'] *= scale_y
            bbox['center_x'] *= scale_x
            bbox['center_y'] *= scale_y
            bbox['width'] *= scale_x
            bbox['height'] *= scale_y

            scaled_detection['bbox'] = bbox
            scaled_detections.append(scaled_detection)

        return scaled_detections
