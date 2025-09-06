#!/bin/bash
#
# HAILO AI HAT Installation Script
# Automated setup for HAILO AI HAT development environment on Raspberry Pi 5
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="hailo-rpi5"
VENV_NAME="hailo-venv"
PYTHON_VERSION="3.9"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_platform() {
    log_info "Checking platform compatibility..."

    if [[ ! -f /proc/device-tree/model ]]; then
        log_warning "Cannot detect device model. Proceeding anyway..."
        return 0
    fi

    MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")
    if [[ "$MODEL" == *"Raspberry Pi 5"* ]]; then
        log_success "Detected Raspberry Pi 5"
    else
        log_warning "Not running on Raspberry Pi 5. Model: $MODEL"
        log_warning "This project is optimized for Raspberry Pi 5 with HAILO AI HAT"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled"
            exit 0
        fi
    fi
}

# Check system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."

    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_info "Found Python $PYTHON_VER"

        # Check if Python version is compatible
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Python version is compatible (>= 3.8)"
        else
            log_error "Python 3.8 or higher is required. Found: $PYTHON_VER"
            exit 1
        fi
    else
        log_error "Python 3 is not installed"
        log_info "Please install Python 3.8+ before running this script"
        exit 1
    fi

    # Check for required system packages
    local missing_packages=()

    if ! command -v git &> /dev/null; then
        missing_packages+=("git")
    fi

    if ! dpkg -l | grep -q python3-venv; then
        missing_packages+=("python3-venv")
    fi

    if ! dpkg -l | grep -q python3-pip; then
        missing_packages+=("python3-pip")
    fi

    if ! dpkg -l | grep -q python3-dev; then
        missing_packages+=("python3-dev")
    fi

    if ! dpkg -l | grep -q build-essential; then
        missing_packages+=("build-essential")
    fi

    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_warning "Missing system packages: ${missing_packages[*]}"
        log_info "Installing missing packages..."

        if [[ $EUID -eq 0 ]]; then
            apt update && apt install -y "${missing_packages[@]}"
        else
            sudo apt update && sudo apt install -y "${missing_packages[@]}"
        fi

        log_success "System packages installed"
    else
        log_success "All required system packages are installed"
    fi
}

# Create and setup virtual environment
setup_virtual_environment() {
    log_info "Setting up Python virtual environment..."

    cd "$SCRIPT_DIR"

    # Remove existing venv if it exists
    if [ -d "$VENV_NAME" ]; then
        log_warning "Existing virtual environment found. Removing..."
        rm -rf "$VENV_NAME"
    fi

    # Create new virtual environment
    python3 -m venv "$VENV_NAME"

    # Activate virtual environment
    source "$VENV_NAME/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    log_success "Virtual environment created and activated"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    # Ensure we're in the virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_error "Virtual environment not activated"
        exit 1
    fi

    # Install basic requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing requirements from requirements.txt..."
        pip install -r requirements.txt
    else
        log_warning "requirements.txt not found. Installing basic packages..."
        pip install numpy pillow opencv-python matplotlib pytest
    fi

    log_success "Python dependencies installed"
}

# Setup HAILO dependencies
setup_hailo_dependencies() {
    log_info "Setting up HAILO dependencies..."

    # Check if PyHailoRT is available
    if python -c "import hailo_platform.pyhailort" 2>/dev/null; then
        log_success "PyHailoRT is already installed"
    else
        log_warning "PyHailoRT not found"
        log_info "Please install HailoRT and PyHailoRT manually:"
        log_info "1. Download HailoRT from Hailo's developer zone"
        log_info "2. Install the appropriate package for your system"
        log_info "3. Ensure PyHailoRT Python bindings are available"

        read -p "Continue without PyHailoRT? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled. Please install PyHailoRT first."
            exit 1
        fi
    fi
}

# Install project in development mode
install_project() {
    log_info "Installing project in development mode..."

    if [ -f "pyproject.toml" ]; then
        pip install -e .
        log_success "Project installed in development mode"
    elif [ -f "setup.py" ]; then
        pip install -e .
        log_success "Project installed in development mode"
    else
        log_warning "No setup.py or pyproject.toml found. Skipping project installation."
        log_info "You can still use the project by adding it to PYTHONPATH"
    fi
}

# Create models directory
setup_directories() {
    log_info "Setting up project directories..."

    mkdir -p models/{classification,detection,segmentation,pose}
    mkdir -p data/{images,videos,datasets}
    mkdir -p logs
    mkdir -p temp

    log_success "Project directories created"
}

# Run tests to verify installation
run_tests() {
    log_info "Running basic tests to verify installation..."

    # Test basic imports
    if python -c "import sys; sys.path.insert(0, '.'); import src" 2>/dev/null; then
        log_success "Basic imports working"
    else
        log_warning "Basic imports failed. This may be normal if PyHailoRT is not installed."
    fi

    # Run pytest if available
    if command -v pytest &> /dev/null && [ -d "tests" ]; then
        log_info "Running unit tests..."
        if pytest tests/ -v --tb=short; then
            log_success "All tests passed"
        else
            log_warning "Some tests failed. This may be expected without hardware."
        fi
    else
        log_info "Skipping tests (pytest not available or no tests directory)"
    fi
}

# Create activation script
create_activation_script() {
    log_info "Creating activation script..."

    cat > activate_hailo.sh << 'EOF'
#!/bin/bash
# HAILO AI HAT Environment Activation Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="hailo-venv"

if [ -d "$SCRIPT_DIR/$VENV_NAME" ]; then
    echo "Activating HAILO AI HAT environment..."
    source "$SCRIPT_DIR/$VENV_NAME/bin/activate"
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    echo "Environment activated. You can now use the HAILO AI HAT tools."
    echo "To deactivate, run: deactivate"
else
    echo "Error: Virtual environment not found. Please run install.sh first."
    exit 1
fi
EOF

    chmod +x activate_hailo.sh
    log_success "Activation script created (activate_hailo.sh)"
}

# Print usage instructions
print_usage_instructions() {
    echo
    echo "============================================================"
    echo "INSTALLATION COMPLETE"
    echo "============================================================"
    echo
    log_success "HAILO AI HAT development environment is ready!"
    echo
    echo "To get started:"
    echo "  1. Activate the environment:"
    echo "     source activate_hailo.sh"
    echo
    echo "  2. Or manually activate:"
    echo "     source $VENV_NAME/bin/activate"
    echo
    echo "  3. Run examples:"
    echo "     python examples/classification_example.py --help"
    echo "     python examples/detection_example.py --help"
    echo
    echo "  4. Download models (optional):"
    echo "     ./scripts/download_models.sh"
    echo
    echo "  5. Run tests:"
    echo "     pytest tests/"
    echo
    echo "For more information, see README.md"
    echo
}

# Main installation function
main() {
    echo "============================================================"
    echo "HAILO AI HAT INSTALLATION SCRIPT"
    echo "============================================================"
    echo

    log_info "Starting installation process..."

    # Check command line arguments
    SKIP_PLATFORM_CHECK=false
    SKIP_TESTS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-platform-check)
                SKIP_PLATFORM_CHECK=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --skip-platform-check    Skip Raspberry Pi platform check"
                echo "  --skip-tests             Skip running tests after installation"
                echo "  --help, -h               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run installation steps
    if [ "$SKIP_PLATFORM_CHECK" = false ]; then
        check_platform
    fi

    check_dependencies
    setup_virtual_environment
    install_dependencies
    setup_hailo_dependencies
    install_project
    setup_directories

    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    fi

    create_activation_script
    print_usage_instructions

    log_success "Installation completed successfully!"
}

# Trap errors
trap 'log_error "Installation failed at line $LINENO"' ERR

# Run main function
main "$@"
