# HAILO AI HAT for Raspberry Pi 5 - Development Project

This project provides a comprehensive development framework for building AI applications using the HAILO AI HAT on Raspberry Pi 5 with LibHailoRT and PyHailoRT APIs.

## Features

- **Device Management**: Easy discovery, initialization, and management of HAILO AI HAT devices
- **Inference Engine**: High-level interface for running neural network inference
- **Preprocessing Utilities**: Common image preprocessing functions with support for various model formats
- **Postprocessing Utilities**: Tools for processing model outputs (classification, detection, segmentation)
- **Model Analysis**: Utilities for analyzing HEF model files and extracting metadata
- **Performance Tools**: Benchmarking and profiling tools for optimization
- **Examples**: Ready-to-run examples for common use cases
- **Automated Installation**: One-command setup with virtual environment and dependency management

## Project Structure

```
hailo-rpi5/
├── src/                        # Main package (renamed from hailo_rpi5)
│   ├── core/                   # Core functionality
│   │   ├── device_manager.py   # Device management
│   │   └── inference_engine.py # Inference engine
│   ├── utils/                  # Utility modules
│   │   ├── model_utils.py      # Model analysis tools
│   │   ├── preprocessing.py    # Image preprocessing
│   │   └── postprocessing.py   # Output postprocessing
│   └── __init__.py
├── examples/                   # Usage examples
│   ├── classification_example.py
│   └── detection_example.py
├── tools/                      # Command-line tools
│   ├── device_info.py         # Device information tool
│   ├── model_analyzer.py      # Model analysis tool
│   └── benchmark.py           # Performance benchmark
├── scripts/                    # Utility scripts
│   └── download_models.sh     # Model download script
├── tests/                      # Unit tests
├── config/                     # Configuration files
├── install.sh                 # Automated installation script
├── activate_hailo.sh          # Environment activation script (generated)
├── requirements.txt           # Python dependencies
└── pyproject.toml            # Project configuration
```

## Installation

### Prerequisites

1. **Raspberry Pi 5** with HAILO AI HAT properly installed
2. **HAILO drivers** installed and configured
3. **Python 3.8+**

### Automated Installation (Recommended)

The easiest way to set up the development environment is using the provided installation script:

```bash
# Clone the repository
git clone <your-repo-url> hailo-rpi5
cd hailo-rpi5

# Run the automated installation script
./install.sh
```

The installation script will:
- Check platform compatibility (Raspberry Pi 5)
- Verify and install system dependencies
- Create a Python virtual environment
- Install all Python dependencies
- Set up project directories
- Create an activation script for easy environment management
- Run basic tests to verify the installation

**Installation Script Options:**
```bash
./install.sh --help                    # Show help
./install.sh --skip-platform-check     # Skip Raspberry Pi detection
./install.sh --skip-tests             # Skip running tests
```

**Activating the Environment:**
After installation, activate the environment using:
```bash
# Use the generated activation script (recommended)
source activate_hailo.sh

# Or manually activate the virtual environment
source hailo-venv/bin/activate
```

### Manual Installation

If you prefer to install manually or need custom configuration:

```bash
# Clone the repository
git clone <your-repo-url> hailo-rpi5
cd hailo-rpi5

# Create virtual environment
python3 -m venv hailo-venv
source hailo-venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Install HAILO Platform

The project requires PyHailoRT (HAILO's Python API). Install it according to HAILO's documentation:

```bash
# This is typically provided by HAILO
pip install hailo-platform
```

**Note:** PyHailoRT is provided by Hailo and requires proper HAILO AI HAT hardware and drivers. The project will work in mock mode for development without the hardware.

## Quick Start

**Important:** Make sure to activate your environment first:
```bash
# If you used the automated installation
source activate_hailo.sh

# Or manually activate if you installed manually
source hailo-venv/bin/activate
```

### 1. Download Models (Optional)

```bash
# Download sample models
./scripts/download_models.sh

# List available models in the project
find models/ -name "*.hef" -type f
```

### 2. Check Device Connectivity

```bash
# Check if HAILO devices are detected
python tools/device_info.py

# Get detailed device information
python tools/device_info.py --hef path/to/your/model.hef
```

### 3. Analyze a Model

```bash
# Analyze HEF model file
python tools/model_analyzer.py path/to/your/model.hef

# Save analysis to JSON
python tools/model_analyzer.py path/to/your/model.hef --output model_analysis.json

# Validate model compatibility
python tools/model_analyzer.py path/to/your/model.hef --validate
```

### 4. Run Classification Example

```bash
# Run image classification
python examples/classification_example.py \
    --hef path/to/classification_model.hef \
    --image path/to/test_image.jpg \
    --top-k 5
```

### 5. Run Object Detection Example

```bash
# Run object detection
python examples/detection_example.py \
    --hef path/to/detection_model.hef \
    --image path/to/test_image.jpg \
    --confidence 0.5 \
    --iou 0.5
```

### 6. Benchmark Performance

```bash
# Run performance benchmark
python tools/benchmark.py \
    --hef path/to/your/model.hef \
    --latency-iterations 100 \
    --throughput-duration 30
```

## Programming Examples

### Basic Device Usage

```python
from hailo_rpi5 import HailoDeviceManager, HailoInferenceEngine

# Initialize device
with HailoDeviceManager() as device_manager:
    # Discover devices
    devices = device_manager.discover_devices()
    print(f"Found {len(devices)} devices")
    
    # Initialize first device
    device_manager.initialize_device()
    
    # Load model
    device_manager.load_hef("model.hef")
    device_manager.configure_network_group()
    
    # Get device info
    info = device_manager.get_device_info()
    print(f"Device: {info}")
```

### Running Inference

```python
from hailo_rpi5 import HailoDeviceManager, HailoInferenceEngine
from hailo_rpi5.utils import PreprocessingUtils
import numpy as np

with HailoDeviceManager() as device_manager:
    # Setup device and model
    device_manager.initialize_device()
    device_manager.load_hef("model.hef")
    device_manager.configure_network_group()
    
    # Setup inference
    engine = HailoInferenceEngine(device_manager)
    engine.setup_inference_pipeline()
    
    # Preprocess image
    config = PreprocessingUtils.get_common_preprocessing_configs()['imagenet']
    preprocessing_fn = PreprocessingUtils.create_preprocessing_pipeline(config)
    processed_image = preprocessing_fn("image.jpg")
    
    # Run inference
    inputs = {"input_layer_name": processed_image}
    outputs = engine.run_inference(inputs)
    
    print(f"Inference results: {outputs}")
```

### Processing Detection Results

```python
from hailo_rpi5.utils import PostprocessingUtils

# Process YOLO detection outputs
detections = PostprocessingUtils.process_detection_output(
    outputs=model_outputs,
    output_format='yolo',
    confidence_threshold=0.5,
    iou_threshold=0.5,
    class_names=['person', 'car', 'bicycle', ...]
)

# Filter detections
person_detections = PostprocessingUtils.filter_detections_by_class(
    detections, ['person']
)

# Print results
for detection in person_detections:
    print(f"Person detected with confidence {detection['confidence']:.2f}")
    print(f"Bounding box: {detection['bbox']}")
```

## Configuration

The project uses YAML configuration files in the `config/` directory:

```yaml
# config/default.yaml
preprocessing:
  target_size: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  scale: 0.00392156862745098

postprocessing:
  confidence_threshold: 0.5
  iou_threshold: 0.5
  max_detections: 100

inference:
  batch_size: 1
  timeout_ms: 10000
```

## API Reference

### Core Classes

- **`HailoDeviceManager`**: Manages HAILO device lifecycle
- **`HailoInferenceEngine`**: High-level interface for inference

### Utility Classes

- **`ModelUtils`**: Model analysis and validation
- **`PreprocessingUtils`**: Image preprocessing functions  
- **`PostprocessingUtils`**: Output processing functions

### Key Methods

```python
# Device Management
device_manager.discover_devices() -> List[str]
device_manager.initialize_device(device_id=None) -> bool
device_manager.load_hef(hef_path) -> bool
device_manager.configure_network_group() -> bool

# Inference
engine.setup_inference_pipeline() -> bool
engine.run_inference(inputs) -> Dict[str, np.ndarray]
engine.benchmark_inference(inputs, iterations) -> Dict

# Preprocessing
PreprocessingUtils.load_and_preprocess_image(path, target_size, ...)
PreprocessingUtils.create_preprocessing_pipeline(config)

# Postprocessing
PostprocessingUtils.process_detection_output(outputs, format, ...)
PostprocessingUtils.get_top_k_predictions(probabilities, k, class_names)
```

## Testing

The project uses **pytest** for modern, comprehensive testing. Tests are organized into unit and integration categories.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run only unit tests (fast, no hardware required)
pytest tests/unit/

# Run integration tests (requires HAILO hardware)
pytest tests/integration/

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=html

# Run specific test files or classes
pytest tests/unit/test_device_manager.py
pytest tests/unit/test_inference_engine.py::TestHailoInferenceEngine
```

### Test Categories

- **Unit Tests** (`tests/unit/`): Fast tests that don't require hardware
  - Device manager functionality
  - Inference engine logic
  - Utility functions
  - Preprocessing/postprocessing

- **Integration Tests** (`tests/integration/`): Tests requiring actual HAILO hardware
  - End-to-end inference pipeline
  - Hardware communication
  - Performance benchmarks

### Make Commands

For convenience, you can also use:

```bash
make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-coverage     # Run tests with coverage report
```

## Troubleshooting

### Common Issues

1. **No devices found**
   - Check HAILO AI HAT hardware connection
   - Verify HAILO drivers are installed
   - Run with `sudo` if needed for device access

2. **PyHailoRT import error**
   - Install HAILO platform package
   - Check Python path and virtual environment

3. **Model loading fails**
   - Verify HEF file path and permissions
   - Check model compatibility with device architecture
   - Use `model_analyzer.py` to validate the model

4. **Inference errors**
   - Check input data shapes and formats
   - Verify preprocessing pipeline matches model requirements
   - Monitor device memory usage

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Troubleshooting

### Installation Issues

**Installation script fails:**
```bash
# Check system requirements
python3 --version  # Should be 3.8+
cat /proc/device-tree/model  # Should show Raspberry Pi 5

# Run with verbose output
./install.sh --skip-tests

# Check log for specific errors
```

**Virtual environment issues:**
```bash
# Remove existing environment and reinstall
rm -rf hailo-venv
./install.sh
```

**PyHailoRT not found:**
```bash
# Verify HAILO drivers are installed
ls /opt/hailo/  # Should contain HailoRT installation

# Check if PyHailoRT is available in system Python
python3 -c "import hailo_platform.pyhailort"

# If not available, install manually or contact HAILO support
```

**Permission errors:**
```bash
# Ensure user has access to HAILO devices
sudo usermod -a -G hailo $USER
# Log out and back in for group changes to take effect
```

### Runtime Issues

**Device not found:**
- Ensure HAILO AI HAT is properly connected
- Verify drivers are loaded: `lsmod | grep hailo`
- Check device permissions: `ls -la /dev/hailo*`

**Model loading fails:**
- Verify HEF file integrity
- Check model compatibility with `python tools/model_analyzer.py --validate`
- Ensure sufficient memory is available

**Performance issues:**
- Monitor system resources: `htop`, `free -h`
- Check thermal throttling: `vcgencmd measure_temp`
- Use performance governor: `sudo cpufreq-set -g performance`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **HAILO Community Forum**: [community.hailo.ai](https://community.hailo.ai/)
- **HAILO Documentation**: [hailo.ai/developer-zone](https://hailo.ai/developer-zone/)
- **Issues**: Create an issue in this repository

## Acknowledgments

- HAILO AI for the HAILO AI HAT and HailoRT framework
- Raspberry Pi Foundation for the Raspberry Pi 5 platform
- The open-source community for various tools and libraries used