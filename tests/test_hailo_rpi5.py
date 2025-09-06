"""
Legacy test file - Tests have been migrated to pytest format.

This file is kept for backward compatibility but the actual tests
are now in the unit/ and integration/ directories using pytest.

To run the new tests:
    pytest tests/unit/           # Unit tests
    pytest tests/integration/    # Integration tests (requires hardware)
    pytest tests/               # All tests
"""

# This file is deprecated - use pytest tests instead
import warnings

warnings.warn(
    "test_hailo_rpi5.py is deprecated. Use 'pytest tests/' to run the new test suite.",
    DeprecationWarning,
    stacklevel=2
)


class TestDeviceManager(unittest.TestCase):
    """Test cases for HailoDeviceManager."""

    @patch('hailo_rpi5.core.device_manager.HAILO_AVAILABLE', True)
    @patch('hailo_rpi5.core.device_manager.Device')
    def test_discover_devices(self, mock_device_class):
        """Test device discovery."""
        # Mock device scanning
        mock_device_class.scan.return_value = [Mock(device_id='device_1')]

        manager = HailoDeviceManager()
        devices = manager.discover_devices()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0], 'device_1')

    @patch('hailo_rpi5.core.device_manager.HAILO_AVAILABLE', False)
    def test_hailo_not_available(self):
        """Test behavior when HAILO is not available."""
        with self.assertRaises(RuntimeError):
            HailoDeviceManager()

    def test_context_manager(self):
        """Test context manager functionality."""
        with patch('hailo_rpi5.core.device_manager.HAILO_AVAILABLE', True):
            manager = HailoDeviceManager()
            manager.cleanup = Mock()

            with manager as mgr:
                self.assertIs(mgr, manager)

            manager.cleanup.assert_called_once()


class TestInferenceEngine(unittest.TestCase):
    """Test cases for HailoInferenceEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_device_manager = Mock(spec=HailoDeviceManager)

    @patch('hailo_rpi5.core.inference_engine.HAILO_AVAILABLE', True)
    def test_initialization(self):
        """Test inference engine initialization."""
        engine = HailoInferenceEngine(self.mock_device_manager)
        self.assertIs(engine.device_manager, self.mock_device_manager)
        self.assertIsNone(engine.infer_pipeline)

    @patch('hailo_rpi5.core.inference_engine.HAILO_AVAILABLE', False)
    def test_hailo_not_available(self):
        """Test behavior when HAILO is not available."""
        with self.assertRaises(RuntimeError):
            HailoInferenceEngine(self.mock_device_manager)


class TestPreprocessingUtils(unittest.TestCase):
    """Test cases for PreprocessingUtils."""

    def test_normalize_image(self):
        """Test image normalization."""
        # Create test image
        image = np.ones((10, 10, 3), dtype=np.uint8) * 128

        # Test simple normalization
        normalized = PreprocessingUtils.normalize_image(
            image, mean=0.5, std=0.5, scale=1.0/255.0
        )

        # Expected result: (128/255 - 0.5) / 0.5 = 0.0039
        expected = np.full_like(image, 0.003921568, dtype=np.float32)
        np.testing.assert_allclose(normalized, expected, rtol=1e-6)

    def test_add_batch_dimension(self):
        """Test adding batch dimension."""
        data = np.random.rand(224, 224, 3)
        batched = PreprocessingUtils.add_batch_dimension(data)

        self.assertEqual(batched.shape, (1, 224, 224, 3))

    def test_convert_to_model_format(self):
        """Test format conversion."""
        image = np.random.rand(224, 224, 3)

        # Test NHWC format
        nhwc = PreprocessingUtils.convert_to_model_format(image, 'NHWC')
        self.assertEqual(nhwc.shape, (1, 224, 224, 3))

        # Test NCHW format
        nchw = PreprocessingUtils.convert_to_model_format(image, 'NCHW')
        self.assertEqual(nchw.shape, (1, 3, 224, 224))

    def test_get_common_preprocessing_configs(self):
        """Test common preprocessing configurations."""
        configs = PreprocessingUtils.get_common_preprocessing_configs()

        self.assertIn('imagenet', configs)
        self.assertIn('yolo', configs)
        self.assertIn('simple', configs)

        # Check ImageNet config structure
        imagenet_config = configs['imagenet']
        self.assertIn('target_size', imagenet_config)
        self.assertIn('mean', imagenet_config)
        self.assertIn('std', imagenet_config)


class TestPostprocessingUtils(unittest.TestCase):
    """Test cases for PostprocessingUtils."""

    def test_softmax(self):
        """Test softmax function."""
        logits = np.array([1.0, 2.0, 3.0])
        probabilities = PostprocessingUtils.softmax(logits)

        # Check if probabilities sum to 1
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=6)

        # Check if probabilities are positive
        self.assertTrue(np.all(probabilities > 0))

    def test_get_top_k_predictions(self):
        """Test top-k predictions."""
        probabilities = np.array([0.1, 0.3, 0.6])
        class_names = ['class_0', 'class_1', 'class_2']

        top_k = PostprocessingUtils.get_top_k_predictions(
            probabilities, k=2, class_names=class_names
        )

        self.assertEqual(len(top_k), 2)
        self.assertEqual(top_k[0]['class_name'], 'class_2')
        self.assertEqual(top_k[0]['rank'], 1)
        self.assertAlmostEqual(top_k[0]['probability'], 0.6)

    def test_nms(self):
        """Test Non-Maximum Suppression."""
        # Create test boxes and scores
        boxes = np.array([
            [10, 10, 50, 50],  # Box 1
            [15, 15, 55, 55],  # Box 2 (overlaps with Box 1)
            [100, 100, 140, 140]  # Box 3 (separate)
        ])
        scores = np.array([0.9, 0.8, 0.7])

        keep_indices = PostprocessingUtils.nms(
            boxes, scores, score_threshold=0.5, iou_threshold=0.5
        )

        # Should keep box 1 (highest score) and box 3 (no overlap)
        self.assertIn(0, keep_indices)  # Box 1
        self.assertIn(2, keep_indices)  # Box 3
        self.assertEqual(len(keep_indices), 2)

    def test_filter_detections_by_class(self):
        """Test filtering detections by class."""
        detections = [
            {'class_name': 'person', 'class_id': 0, 'confidence': 0.9},
            {'class_name': 'car', 'class_id': 1, 'confidence': 0.8},
            {'class_name': 'dog', 'class_id': 2, 'confidence': 0.7}
        ]

        # Filter by class names
        filtered = PostprocessingUtils.filter_detections_by_class(
            detections, ['person', 'car']
        )

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['class_name'], 'person')
        self.assertEqual(filtered[1]['class_name'], 'car')


class TestModelUtils(unittest.TestCase):
    """Test cases for ModelUtils."""

    @patch('hailo_rpi5.utils.model_utils.HAILO_AVAILABLE', False)
    def test_hailo_not_available(self):
        """Test behavior when HAILO is not available."""
        with self.assertRaises(RuntimeError):
            ModelUtils.analyze_hef('dummy_path.hef')

    def test_list_model_files(self):
        """Test listing model files."""
        # This test would require actual file system setup
        # For now, test with non-existent directory
        models = ModelUtils.list_model_files('/non/existent/path')
        self.assertEqual(len(models), 0)

    def test_estimate_inference_memory(self):
        """Test memory estimation."""
        analysis = {
            'file_size_mb': 10.0,
            'input_streams': [
                {'shape': [224, 224, 3], 'name': 'input1'}
            ],
            'output_streams': [
                {'shape': [1000], 'name': 'output1'}
            ]
        }

        estimates = ModelUtils.estimate_inference_memory(analysis)

        self.assertIn('model_size_mb', estimates)
        self.assertIn('input_buffers_mb', estimates)
        self.assertIn('output_buffers_mb', estimates)
        self.assertIn('total_estimated_mb', estimates)

        # Check that total is sum of components
        expected_total = (
            estimates['model_size_mb'] +
            estimates['input_buffers_mb'] +
            estimates['output_buffers_mb'] +
            50  # overhead
        )
        self.assertAlmostEqual(estimates['total_estimated_mb'], expected_total)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
