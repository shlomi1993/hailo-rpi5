"""
Unit tests for utility modules.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.model_utils import ModelUtils
from src.utils.preprocessing import PreprocessingUtils
from src.utils.postprocessing import PostprocessingUtils


class TestModelUtils:
    """Test cases for ModelUtils."""

    def test_validate_model_path_valid(self, tmp_path):
        """Test validation of valid model path."""
        # Create a temporary .hef file
        model_file = tmp_path / "test_model.hef"
        model_file.write_text("dummy hef content")
        
        result = ModelUtils.validate_model_path(str(model_file))
        assert result is True

    def test_validate_model_path_invalid_extension(self, tmp_path):
        """Test validation of model with invalid extension."""
        model_file = tmp_path / "test_model.txt"
        model_file.write_text("dummy content")
        
        result = ModelUtils.validate_model_path(str(model_file))
        assert result is False

    def test_validate_model_path_nonexistent(self):
        """Test validation of nonexistent model path."""
        result = ModelUtils.validate_model_path("/nonexistent/path/model.hef")
        assert result is False

    def test_get_model_info(self, tmp_path):
        """Test getting model information."""
        model_file = tmp_path / "test_model.hef"
        model_file.write_text("dummy hef content")
        
        info = ModelUtils.get_model_info(str(model_file))
        assert 'path' in info
        assert 'size' in info
        assert 'exists' in info
        assert info['exists'] is True


class TestPreprocessingUtils:
    """Test cases for PreprocessingUtils."""

    def test_resize_image(self):
        """Test image resizing."""
        # Create a sample image
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        target_size = (224, 224)
        
        resized = PreprocessingUtils.resize_image(image, target_size)
        
        assert resized.shape[:2] == target_size
        assert resized.dtype == np.uint8

    def test_normalize_image(self):
        """Test image normalization."""
        image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        
        normalized = PreprocessingUtils.normalize_image(image)
        
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    def test_preprocess_image_full_pipeline(self):
        """Test full preprocessing pipeline."""
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        target_shape = (224, 224, 3)
        
        processed = PreprocessingUtils.preprocess_image(image, target_shape)
        
        assert processed.shape == target_shape
        # Should be normalized
        assert processed.dtype == np.float32
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)

    def test_center_crop(self):
        """Test center cropping."""
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        crop_size = (300, 300)
        
        cropped = PreprocessingUtils.center_crop(image, crop_size)
        
        assert cropped.shape[:2] == crop_size
        assert cropped.dtype == np.uint8

    def test_pad_image(self):
        """Test image padding."""
        image = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)
        target_size = (224, 224)
        
        padded = PreprocessingUtils.pad_image(image, target_size)
        
        assert padded.shape[:2] == target_size
        assert padded.dtype == np.uint8


class TestPostprocessingUtils:
    """Test cases for PostprocessingUtils."""

    def test_apply_nms(self):
        """Test Non-Maximum Suppression."""
        # Create sample bounding boxes and scores
        boxes = np.array([
            [10, 10, 50, 50],
            [15, 15, 55, 55],  # Overlapping box
            [100, 100, 140, 140]  # Non-overlapping box
        ], dtype=np.float32)
        
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        
        keep_indices = PostprocessingUtils.apply_nms(boxes, scores, threshold=0.5)
        
        # Should keep the best box from overlapping pair and the non-overlapping box
        assert len(keep_indices) == 2
        assert 0 in keep_indices  # Best overlapping box
        assert 2 in keep_indices  # Non-overlapping box

    def test_filter_by_confidence(self):
        """Test confidence-based filtering."""
        detections = [
            {'confidence': 0.9, 'class': 'person'},
            {'confidence': 0.3, 'class': 'car'},  # Below threshold
            {'confidence': 0.7, 'class': 'bicycle'}
        ]
        
        filtered = PostprocessingUtils.filter_by_confidence(detections, threshold=0.5)
        
        assert len(filtered) == 2
        assert all(det['confidence'] >= 0.5 for det in filtered)

    def test_convert_bbox_format(self):
        """Test bounding box format conversion."""
        # Test XYWH to XYXY conversion
        bbox_xywh = [10, 20, 30, 40]  # x, y, width, height
        bbox_xyxy = PostprocessingUtils.convert_bbox_format(bbox_xywh, from_format='xywh', to_format='xyxy')
        
        expected_xyxy = [10, 20, 40, 60]  # x1, y1, x2, y2
        assert bbox_xyxy == expected_xyxy

    def test_scale_bbox(self):
        """Test bounding box scaling."""
        bbox = [10, 20, 30, 40]  # Original coordinates
        original_size = (100, 200)
        target_size = (200, 400)
        
        scaled_bbox = PostprocessingUtils.scale_bbox(bbox, original_size, target_size)
        
        # Should double all coordinates
        expected = [20, 40, 60, 80]
        assert scaled_bbox == expected

    def test_parse_classification_output(self):
        """Test classification output parsing."""
        # Create sample classification output
        output = np.array([0.1, 0.8, 0.05, 0.05], dtype=np.float32)
        class_names = ['cat', 'dog', 'bird', 'fish']
        
        result = PostprocessingUtils.parse_classification_output(output, class_names)
        
        assert result['predicted_class'] == 'dog'
        assert result['confidence'] == 0.8
        assert len(result['top_predictions']) == len(class_names)

    def test_parse_detection_output(self):
        """Test detection output parsing."""
        # Create sample detection output
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        classes = np.array([0, 1], dtype=np.int32)
        class_names = ['person', 'car']
        
        results = PostprocessingUtils.parse_detection_output(
            boxes, scores, classes, class_names, confidence_threshold=0.5
        )
        
        assert len(results) == 2
        assert results[0]['class'] == 'person'
        assert results[0]['confidence'] == 0.9
        assert results[1]['class'] == 'car'
        assert results[1]['confidence'] == 0.7
