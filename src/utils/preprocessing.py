"""
Preprocessing Utilities

Common preprocessing functions for preparing data for HAILO AI HAT inference,
including image resizing, normalization, and format conversion.
"""

from typing import Tuple, Union, Optional, Dict, Any
import logging
import numpy as np
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logging.warning("OpenCV not available")
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logging.warning("PIL not available")
    PIL_AVAILABLE = False


class PreprocessingUtils:
    """Utilities for preprocessing data for HAILO inference."""

    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                    interpolation: str = 'bilinear') -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image as numpy array (H, W, C)
            target_size: Target size as (width, height)
            interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')

        Returns:
            Resized image as numpy array
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for image resizing")

        interpolation_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC
        }

        if interpolation not in interpolation_map:
            raise ValueError(f"Unsupported interpolation: {interpolation}")

        resized = cv2.resize(image, target_size, interpolation=interpolation_map[interpolation])

        # Ensure 3D array (add channel dimension if needed)
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=2)

        return resized

    @staticmethod
    def normalize_image(image: np.ndarray, mean: Union[float, Tuple[float, ...]] = 0.0,
                       std: Union[float, Tuple[float, ...]] = 1.0,
                       scale: float = 1.0) -> np.ndarray:
        """
        Normalize image pixel values.

        Args:
            image: Input image as numpy array
            mean: Mean values for normalization (per channel or single value)
            std: Standard deviation values for normalization (per channel or single value)
            scale: Scale factor applied before normalization

        Returns:
            Normalized image as numpy array
        """
        # Convert to float32 for processing
        normalized = image.astype(np.float32)

        # Apply scaling
        normalized = normalized * scale

        # Apply normalization
        if isinstance(mean, (int, float)):
            normalized = (normalized - mean) / std
        else:
            # Per-channel normalization
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)

            if len(normalized.shape) == 3:
                # Reshape for broadcasting (H, W, C)
                mean = mean.reshape(1, 1, -1)
                std = std.reshape(1, 1, -1)
            elif len(normalized.shape) == 4:
                # Batch dimension (N, H, W, C)
                mean = mean.reshape(1, 1, 1, -1)
                std = std.reshape(1, 1, 1, -1)

            normalized = (normalized - mean) / std

        return normalized

    @staticmethod
    def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int],
                                 mean: Union[float, Tuple[float, ...]] = 0.0,
                                 std: Union[float, Tuple[float, ...]] = 1.0,
                                 scale: float = 1.0/255.0,
                                 color_format: str = 'RGB') -> np.ndarray:
        """
        Load and preprocess an image file.

        Args:
            image_path: Path to image file
            target_size: Target size as (width, height)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            scale: Scale factor
            color_format: Color format ('RGB', 'BGR', 'GRAY')

        Returns:
            Preprocessed image as numpy array
        """
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        if CV2_AVAILABLE:
            image = cv2.imread(str(image_file))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert color format
            if color_format.upper() == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif color_format.upper() == 'GRAY':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=2)  # Add channel dimension
            # BGR is default for OpenCV

        elif PIL_AVAILABLE:
            image = Image.open(image_file)

            if color_format.upper() == 'RGB':
                image = image.convert('RGB')
            elif color_format.upper() == 'BGR':
                image = image.convert('RGB')
                # PIL doesn't directly support BGR, we'll convert later
            elif color_format.upper() == 'GRAY':
                image = image.convert('L')

            image = np.array(image)

            # Convert RGB to BGR if needed
            if color_format.upper() == 'BGR' and len(image.shape) == 3:
                image = image[:, :, ::-1]  # Reverse channel order

            # Add channel dimension for grayscale
            if color_format.upper() == 'GRAY' and len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
        else:
            raise RuntimeError("Neither OpenCV nor PIL is available for image loading")

        # Preprocess
        image = PreprocessingUtils.resize_image(image, target_size)
        image = PreprocessingUtils.normalize_image(image, mean, std, scale)

        return image

    @staticmethod
    def add_batch_dimension(data: np.ndarray) -> np.ndarray:
        """
        Add batch dimension to data.

        Args:
            data: Input data

        Returns:
            Data with batch dimension added
        """
        return np.expand_dims(data, axis=0)

    @staticmethod
    def convert_to_model_format(image: np.ndarray,
                               target_format: str = 'NHWC') -> np.ndarray:
        """
        Convert image to model input format.

        Args:
            image: Input image (H, W, C) or (N, H, W, C)
            target_format: Target format ('NHWC', 'NCHW', 'HWC', 'CHW')

        Returns:
            Converted image
        """
        if target_format == 'NHWC':
            if len(image.shape) == 3:  # (H, W, C) -> (N, H, W, C)
                return PreprocessingUtils.add_batch_dimension(image)
            return image  # Already in NHWC format

        elif target_format == 'NCHW':
            if len(image.shape) == 3:  # (H, W, C) -> (C, H, W) -> (N, C, H, W)
                image = np.transpose(image, (2, 0, 1))
                return PreprocessingUtils.add_batch_dimension(image)
            elif len(image.shape) == 4:  # (N, H, W, C) -> (N, C, H, W)
                return np.transpose(image, (0, 3, 1, 2))
            return image

        elif target_format == 'HWC':
            if len(image.shape) == 4:  # (N, H, W, C) -> (H, W, C)
                return np.squeeze(image, axis=0)
            return image  # Already in HWC format

        elif target_format == 'CHW':
            if len(image.shape) == 3:  # (H, W, C) -> (C, H, W)
                return np.transpose(image, (2, 0, 1))
            elif len(image.shape) == 4:  # (N, H, W, C) -> (C, H, W)
                image = np.squeeze(image, axis=0)
                return np.transpose(image, (2, 0, 1))
            return image

        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    @staticmethod
    def create_preprocessing_pipeline(config: Dict[str, Any]):
        """
        Create a preprocessing pipeline from configuration.

        Args:
            config: Configuration dictionary with preprocessing parameters

        Returns:
            Function that applies the preprocessing pipeline
        """
        def pipeline(image_path: str) -> np.ndarray:
            # Extract parameters from config
            target_size = tuple(config.get('target_size', (224, 224)))
            mean = config.get('mean', 0.0)
            std = config.get('std', 1.0)
            scale = config.get('scale', 1.0/255.0)
            color_format = config.get('color_format', 'RGB')
            model_format = config.get('model_format', 'NHWC')

            # Load and preprocess
            image = PreprocessingUtils.load_and_preprocess_image(
                image_path, target_size, mean, std, scale, color_format
            )

            # Convert to model format
            image = PreprocessingUtils.convert_to_model_format(image, model_format)

            return image

        return pipeline

    @staticmethod
    def batch_preprocess_images(image_paths: list, preprocessing_fn) -> np.ndarray:
        """
        Preprocess multiple images into a batch.

        Args:
            image_paths: List of image file paths
            preprocessing_fn: Preprocessing function to apply to each image

        Returns:
            Batch of preprocessed images
        """
        processed_images = []

        for image_path in image_paths:
            try:
                processed_image = preprocessing_fn(image_path)
                # Remove batch dimension if present (we'll add it back for the entire batch)
                if len(processed_image.shape) == 4 and processed_image.shape[0] == 1:
                    processed_image = np.squeeze(processed_image, axis=0)
                processed_images.append(processed_image)
            except Exception as e:
                logging.error(f"Error preprocessing {image_path}: {e}")
                continue

        if not processed_images:
            raise ValueError("No images were successfully preprocessed")

        # Stack into batch
        batch = np.stack(processed_images, axis=0)
        return batch

    @staticmethod
    def get_common_preprocessing_configs() -> Dict[str, Dict[str, Any]]:
        """
        Get common preprocessing configurations for different model types.

        Returns:
            Dictionary of common preprocessing configurations
        """
        configs = {
            'imagenet': {
                'target_size': (224, 224),
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'scale': 1.0/255.0,
                'color_format': 'RGB',
                'model_format': 'NHWC'
            },
            'mobilenet': {
                'target_size': (224, 224),
                'mean': 0.5,
                'std': 0.5,
                'scale': 1.0/255.0,
                'color_format': 'RGB',
                'model_format': 'NHWC'
            },
            'yolo': {
                'target_size': (640, 640),
                'mean': 0.0,
                'std': 1.0,
                'scale': 1.0/255.0,
                'color_format': 'RGB',
                'model_format': 'NHWC'
            },
            'simple': {
                'target_size': (224, 224),
                'mean': 0.0,
                'std': 1.0,
                'scale': 1.0/255.0,
                'color_format': 'RGB',
                'model_format': 'NHWC'
            }
        }

        return configs
