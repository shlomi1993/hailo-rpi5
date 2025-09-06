#!/bin/bash

# HAILO Model Download Script
# Downloads pre-compiled HEF models for the HAILO AI HAT project

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
TEMP_DIR="${SCRIPT_DIR}/temp_downloads"

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

# Create directories
create_directories() {
    log_info "Creating directories..."
    mkdir -p "$MODELS_DIR"
    mkdir -p "$TEMP_DIR"

    # Create subdirectories for different model types
    mkdir -p "$MODELS_DIR/classification"
    mkdir -p "$MODELS_DIR/detection"
    mkdir -p "$MODELS_DIR/segmentation"
    mkdir -p "$MODELS_DIR/pose"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
    log_error "Neither wget nor curl is available. Please install one of them."
    exit 1
    fi

    if ! command -v unzip &> /dev/null; then
    log_error "unzip is not available. Please install it."
        exit 1
    fi
}

# Download function with retry
download_file() {
    local url="$1"
    local output_path="$2"
    local max_retries=3
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        log_info "Downloading $(basename "$output_path") (attempt $((retry_count + 1))/$max_retries)..."

        if command -v wget &> /dev/null; then
            if wget -q --show-progress --timeout=30 -O "$output_path" "$url"; then
                return 0
            fi
        elif command -v curl &> /dev/null; then
            if curl -L --progress-bar --connect-timeout 30 -o "$output_path" "$url"; then
                return 0
            fi
        fi

        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log_warning "Download failed, retrying in 5 seconds..."
            sleep 5
        fi
    done

    log_error "Failed to download $url after $max_retries attempts"
    return 1
}

# Model definitions
# Note: These are example URLs - replace with actual HAILO model URLs
declare -A MODELS=(
    # Classification models
    ["resnet50_classification"]="classification|https://example.com/models/resnet50.hef|ResNet-50 image classification model"
    ["mobilenet_v2_classification"]="classification|https://example.com/models/mobilenet_v2.hef|MobileNet-v2 classification model"
    ["efficientnet_classification"]="classification|https://example.com/models/efficientnet.hef|EfficientNet classification model"

    # Object detection models
    ["yolov5s_detection"]="detection|https://example.com/models/yolov5s.hef|YOLOv5s object detection model"
    ["yolov8n_detection"]="detection|https://example.com/models/yolov8n.hef|YOLOv8n object detection model"
    ["ssd_mobilenet_detection"]="detection|https://example.com/models/ssd_mobilenet.hef|SSD MobileNet detection model"

    # Segmentation models
    ["deeplabv3_segmentation"]="segmentation|https://example.com/models/deeplabv3.hef|DeepLabv3 segmentation model"
    ["unet_segmentation"]="segmentation|https://example.com/models/unet.hef|U-Net segmentation model"

    # Pose estimation models
    ["openpose_pose"]="pose|https://example.com/models/openpose.hef|OpenPose human pose estimation model"
)

# Alternative: HAILO Model Zoo URLs (these would be the actual URLs from HAILO)
declare -A HAILO_ZOO_MODELS=(
    # Example structure - replace with actual HAILO Model Zoo URLs
    ["resnet50_hailo"]="classification|https://hailo-model-zoo.s3.amazonaws.com/Classification/resnet_v1_50/pretrained/2022-04-19/resnet_v1_50.hef|ResNet-50 from HAILO Model Zoo"
    ["yolov5s_hailo"]="detection|https://hailo-model-zoo.s3.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s/pretrained/2022-04-19/yolov5s.hef|YOLOv5s from HAILO Model Zoo"
    ["mobilenet_hailo"]="classification|https://hailo-model-zoo.s3.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2022-04-19/mobilenet_v2_1.0.hef|MobileNet-v2 from HAILO Model Zoo"
)

# Download a specific model
download_model() {
    local model_name="$1"
    local model_info="${MODELS[$model_name]}"

    IFS='|' read -r category url description <<< "$model_info"

    local filename="$(basename "$url")"
    local target_dir="$MODELS_DIR/$category"
    local target_path="$target_dir/$filename"

    if [ -f "$target_path" ]; then
        log_warning "Model $model_name already exists at $target_path"
        return 0
    fi

    log_info "Downloading $description..."

    if download_file "$url" "$target_path"; then
        # Verify the file is a valid HEF (basic check)
        if file "$target_path" | grep -q "data" || [[ "$filename" == *.hef ]]; then
            log_success "Downloaded $model_name to $target_path"

            # Create a metadata file
            cat > "${target_path%.hef}.info" << EOF
{
    "name": "$model_name",
    "description": "$description",
    "category": "$category",
    "filename": "$filename",
    "downloaded_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "source_url": "$url"
}
EOF
        else
            log_error "Downloaded file doesn't appear to be a valid HEF model"
            rm -f "$target_path"
            return 1
        fi
    else
        return 1
    fi
}

# List available models
list_models() {
    echo -e "\n${BLUE}Available Models:${NC}"
    echo "=================="

    for model_name in "${!MODELS[@]}"; do
        IFS='|' read -r category url description <<< "${MODELS[$model_name]}"
        printf "%-25s %-15s %s\n" "$model_name" "[$category]" "$description"
    done

    echo -e "\n${BLUE}HAILO Model Zoo Models:${NC}"
    echo "======================="

    for model_name in "${!HAILO_ZOO_MODELS[@]}"; do
        IFS='|' read -r category url description <<< "${HAILO_ZOO_MODELS[$model_name]}"
        printf "%-25s %-15s %s\n" "$model_name" "[$category]" "$description"
    done
}

# Download all models in a category
download_category() {
    local target_category="$1"
    local count=0

    log_info "Downloading all $target_category models..."

    for model_name in "${!MODELS[@]}"; do
        IFS='|' read -r category url description <<< "${MODELS[$model_name]}"
        if [ "$category" = "$target_category" ]; then
            download_model "$model_name"
            count=$((count + 1))
        fi
    done

    # Also check HAILO Zoo models
    for model_name in "${!HAILO_ZOO_MODELS[@]}"; do
        IFS='|' read -r category url description <<< "${HAILO_ZOO_MODELS[$model_name]}"
        if [ "$category" = "$target_category" ]; then
            download_hailo_zoo_model "$model_name"
            count=$((count + 1))
        fi
    done

    log_success "Attempted to download $count models in category: $target_category"
}

# Download HAILO Model Zoo model
download_hailo_zoo_model() {
    local model_name="$1"
    local model_info="${HAILO_ZOO_MODELS[$model_name]}"

    IFS='|' read -r category url description <<< "$model_info"

    local filename="$(basename "$url")"
    local target_dir="$MODELS_DIR/$category"
    local target_path="$target_dir/$filename"

    if [ -f "$target_path" ]; then
        log_warning "Model $model_name already exists at $target_path"
        return 0
    fi

    log_info "Downloading $description from HAILO Model Zoo..."

    if download_file "$url" "$target_path"; then
        log_success "Downloaded $model_name to $target_path"

        # Create a metadata file
        cat > "${target_path%.hef}.info" << EOF
{
    "name": "$model_name",
    "description": "$description",
    "category": "$category",
    "filename": "$filename",
    "downloaded_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "source_url": "$url",
    "source": "HAILO Model Zoo"
}
EOF
    else
        return 1
    fi
}

# Download sample models (quick start set)
download_samples() {
    log_info "Downloading sample models for quick start..."

    # Download one model from each category
    local sample_models=(
        "mobilenet_v2_classification"
        "yolov5s_detection"
    )

    for model in "${sample_models[@]}"; do
        if [[ -v "MODELS[$model]" ]]; then
            download_model "$model"
        fi
    done

    # Also try HAILO Zoo samples
    local hailo_samples=(
        "mobilenet_hailo"
        "yolov5s_hailo"
    )

    for model in "${hailo_samples[@]}"; do
        if [[ -v "HAILO_ZOO_MODELS[$model]" ]]; then
            download_hailo_zoo_model "$model"
        fi
    done
}

# Show downloaded models
show_downloaded() {
    echo -e "\n${BLUE}Downloaded Models:${NC}"
    echo "=================="

    if [ ! -d "$MODELS_DIR" ]; then
        log_warning "Models directory doesn't exist yet"
        return
    fi

    find "$MODELS_DIR" -name "*.hef" -type f | while read -r hef_file; do
        local size=$(du -h "$hef_file" | cut -f1)
        local relative_path="${hef_file#$MODELS_DIR/}"
        echo "  $relative_path ($size)"

        # Show info if available
        local info_file="${hef_file%.hef}.info"
        if [ -f "$info_file" ]; then
            local description=$(grep '"description"' "$info_file" | cut -d'"' -f4)
            echo "    Description: $description"
        fi
    done
}

# Cleanup function
cleanup() {
    if [ -d "$TEMP_DIR" ]; then
        log_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi
}

# Trap to ensure cleanup
trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
HAILO Model Download Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    list                    List all available models
    download MODEL_NAME     Download a specific model
    download-category CAT   Download all models in a category (classification, detection, segmentation, pose)
    download-samples        Download a set of sample models for quick start
    download-all           Download all available models
    show                   Show already downloaded models
    clean                  Remove all downloaded models

Options:
    -h, --help             Show this help message
    --models-dir DIR       Specify custom models directory (default: ./models)

Examples:
    $0 list
    $0 download resnet50_classification
    $0 download-category detection
    $0 download-samples
    $0 show

Note: This script downloads models for the HAILO AI HAT. Make sure you have
sufficient storage space and a stable internet connection.

EOF
}

# Main function
main() {
    local models_dir_override=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --models-dir)
                models_dir_override="$2"
                shift 2
                ;;
            list)
                list_models
                exit 0
                ;;
            download)
                if [ -z "${2:-}" ]; then
                    log_error "Please specify a model name"
                    exit 1
                fi
                check_dependencies
                create_directories
                if [[ -v "MODELS[$2]" ]]; then
                    download_model "$2"
                elif [[ -v "HAILO_ZOO_MODELS[$2]" ]]; then
                    download_hailo_zoo_model "$2"
                else
                    log_error "Unknown model: $2"
                    log_info "Run '$0 list' to see available models"
                    exit 1
                fi
                exit 0
                ;;
            download-category)
                if [ -z "${2:-}" ]; then
                    log_error "Please specify a category"
                    exit 1
                fi
                check_dependencies
                create_directories
                download_category "$2"
                exit 0
                ;;
            download-samples)
                check_dependencies
                create_directories
                download_samples
                exit 0
                ;;
            download-all)
                check_dependencies
                create_directories
                log_info "Downloading all models..."
                for model in "${!MODELS[@]}"; do
                    download_model "$model"
                done
                for model in "${!HAILO_ZOO_MODELS[@]}"; do
                    download_hailo_zoo_model "$model"
                done
                exit 0
                ;;
            show)
                show_downloaded
                exit 0
                ;;
            clean)
                log_warning "This will remove all downloaded models. Are you sure? (y/N)"
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    rm -rf "$MODELS_DIR"
                    log_success "All models removed"
                else
                    log_info "Operation cancelled"
                fi
                exit 0
                ;;
            *)
                log_error "Unknown command: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # If no command provided, show help
    show_help
}

# Override models directory if specified
if [ -n "${models_dir_override:-}" ]; then
    MODELS_DIR="$models_dir_override"
fi

# Run main function
main "$@"
