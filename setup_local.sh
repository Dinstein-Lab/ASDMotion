#!/bin/bash
set -e

# =============================================================================
# ASDMotion Local Setup Script
# =============================================================================
# This script sets up ASDMotion and its dependencies for local testing.
# It creates two conda environments:
#   1. asdmotion    - Main environment for running the pipeline
#   2. open-mmlab   - Separate environment for MMAction2 (PoseC3D model)
#
# Prerequisites:
#   - Anaconda or Miniconda installed
#   - NVIDIA GPU with CUDA support (recommended, CPU fallback available)
#   - ~10GB free disk space
#   - ffmpeg installed (sudo apt install ffmpeg)
#
# Usage:
#   chmod +x setup_local.sh
#   ./setup_local.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCES_DIR="$SCRIPT_DIR/resources"
MODELS_DIR="$RESOURCES_DIR/models"
DEPS_DIR="$SCRIPT_DIR/deps"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- Check prerequisites ----
check_prerequisites() {
    info "Checking prerequisites..."

    if ! command -v conda &>/dev/null; then
        error "Conda is not installed. Please install Miniconda first:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh"
        exit 1
    fi

    if ! command -v ffmpeg &>/dev/null; then
        warn "ffmpeg is not installed. Installing..."
        if command -v apt &>/dev/null; then
            sudo apt update && sudo apt install -y ffmpeg
        elif command -v yum &>/dev/null; then
            sudo yum install -y ffmpeg
        elif command -v brew &>/dev/null; then
            brew install ffmpeg
        else
            error "Cannot install ffmpeg automatically. Please install it manually."
            exit 1
        fi
    fi

    if command -v nvidia-smi &>/dev/null; then
        info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        GPU_AVAILABLE=true
    else
        warn "No NVIDIA GPU detected. The system will run on CPU (much slower)."
        GPU_AVAILABLE=false
    fi

    info "Prerequisites check complete."
}

# ---- Create directories ----
create_directories() {
    info "Creating required directories..."
    mkdir -p "$MODELS_DIR"
    mkdir -p "$RESOURCES_DIR/logs"
    mkdir -p "$RESOURCES_DIR/runs"
    mkdir -p "$DEPS_DIR"
    info "Directories created."
}

# ---- Setup main ASDMotion environment ----
setup_asdmotion_env() {
    info "Setting up 'asdmotion' conda environment..."

    if conda env list | grep -q "^asdmotion "; then
        warn "Environment 'asdmotion' already exists. Skipping creation."
    else
        conda create -n asdmotion python=3.9 -y
    fi

    eval "$(conda shell.bash hook)"
    conda activate asdmotion

    info "Installing Python dependencies..."
    pip install -r "$SCRIPT_DIR/requirements_local.txt"

    # Install mmcv (required by preprocess.py for Config loading)
    if [ "$GPU_AVAILABLE" = true ]; then
        pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html 2>/dev/null || \
        pip install mmcv==2.0.0 2>/dev/null || \
        pip install openmim && mim install mmcv-full 2>/dev/null || \
        warn "Could not install mmcv-full with CUDA. Trying CPU version..."
        pip install mmcv
    else
        pip install mmcv
    fi

    # Install Child Detector (optional but recommended)
    info "Installing Child Detector..."
    pip install git+https://github.com/TalBarami/Child-Detector.git || \
        warn "Child Detector installation failed. You can still run without child detection."

    # Install SkeletonTools
    info "Installing SkeletonTools..."
    pip install git+https://github.com/TalBarami/SkeletonTools.git || \
        warn "SkeletonTools installation failed."

    conda deactivate
    info "ASDMotion environment setup complete."
}

# ---- Setup MMAction2 environment ----
setup_mmaction_env() {
    info "Setting up 'open-mmlab' conda environment for MMAction2..."

    if conda env list | grep -q "^open-mmlab "; then
        warn "Environment 'open-mmlab' already exists. Skipping creation."
    else
        conda create -n open-mmlab python=3.9 -y
    fi

    eval "$(conda shell.bash hook)"
    conda activate open-mmlab

    # Install PyTorch
    if [ "$GPU_AVAILABLE" = true ]; then
        info "Installing PyTorch with CUDA support..."
        pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
    else
        info "Installing PyTorch (CPU only)..."
        pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
    fi

    # Install mmcv and mmaction2
    info "Installing OpenMMLab dependencies..."
    pip install openmim
    mim install mmcv-full==1.7.1 2>/dev/null || pip install mmcv-full || pip install mmcv

    # Clone the forked MMAction2 repository
    MMACTION_DIR="$DEPS_DIR/mmaction2"
    if [ -d "$MMACTION_DIR" ]; then
        warn "MMAction2 already cloned at $MMACTION_DIR"
    else
        info "Cloning MMAction2 (forked repository)..."
        git clone https://github.com/TalBarami/mmaction2.git "$MMACTION_DIR"
    fi

    cd "$MMACTION_DIR"
    pip install -e .
    cd "$SCRIPT_DIR"

    conda deactivate
    info "MMAction2 environment setup complete."
}

# ---- Setup OpenPose ----
setup_openpose() {
    info "=== OpenPose Setup ==="
    OPENPOSE_DIR="$DEPS_DIR/openpose"

    if [ -d "$OPENPOSE_DIR" ] && [ -f "$OPENPOSE_DIR/bin/OpenPoseDemo.exe" -o -f "$OPENPOSE_DIR/build/examples/openpose/openpose.bin" ]; then
        warn "OpenPose already exists at $OPENPOSE_DIR"
        return
    fi

    echo ""
    warn "OpenPose requires manual installation (it needs CMake + CUDA to build)."
    echo ""
    echo "  Option A: Build from source (recommended for Linux):"
    echo "    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git $OPENPOSE_DIR"
    echo "    cd $OPENPOSE_DIR"
    echo "    mkdir build && cd build"
    echo "    cmake .. -DBUILD_PYTHON=OFF -DDOWNLOAD_BODY_25_MODEL=ON"
    echo "    make -j\$(nproc)"
    echo ""
    echo "  Option B: Use pre-built portable version (Windows only):"
    echo "    Download from: https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases"
    echo ""
    echo "  Option C: Use Docker (easiest for Linux):"
    echo "    See LOCAL_SETUP.md for Docker-based OpenPose instructions."
    echo ""
    echo "  After installation, update the config.yaml with the OpenPose path."
    echo ""
}

# ---- Download model weights ----
download_models() {
    info "=== Model Weights ==="

    echo ""
    echo "You need to download the following model files manually from Google Drive:"
    echo ""
    echo "  1. ASDMotion checkpoint (required for inference):"
    echo "     URL: https://drive.google.com/file/d/1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE/view"
    echo "     Save to: $MODELS_DIR/asdmotion.pth"
    echo ""
    echo "  2. Child Detector model (optional, for multi-person videos):"
    echo "     This should be auto-downloaded by YOLOv5, but if needed:"
    echo "     Save to: $MODELS_DIR/child_detector.pt"
    echo ""
    echo "  3. Dataset (optional, for testing):"
    echo "     URL: https://drive.google.com/file/d/1MiNIhlf4mL-vRW1ub2TP3nCYzfMW0bYt/view"
    echo ""

    # Try using gdown if available
    if command -v gdown &>/dev/null || pip install gdown 2>/dev/null; then
        read -p "Attempt automatic download of ASDMotion checkpoint? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Downloading ASDMotion checkpoint..."
            gdown "1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE" -O "$MODELS_DIR/asdmotion.pth" || \
                warn "Auto-download failed. Please download manually from the URL above."
        fi
    else
        warn "gdown not available. Please download model files manually."
    fi
}

# ---- Update configuration ----
update_config() {
    info "Updating configuration file..."

    MMACTION_DIR="$DEPS_DIR/mmaction2"
    OPENPOSE_DIR="$DEPS_DIR/openpose"
    MMLAB_PYTHON="$(conda env list | grep open-mmlab | awk '{print $NF}')/bin/python"

    CONFIG_FILE="$RESOURCES_DIR/configs/config.yaml"
    CONFIG_LOCAL="$RESOURCES_DIR/configs/config_local.yaml"

    cat > "$CONFIG_LOCAL" <<EOF
sequence_length: 200
step_size: 30
model_name: 'asdmotion'
child_detection: true
classification_threshold: 0.85
num_person_in: 5
num_person_out: 5
open_pose_path: '$OPENPOSE_DIR'
mmaction_path: '$MMACTION_DIR'
mmlab_python_path: '$MMLAB_PYTHON'
EOF

    info "Local configuration written to: $CONFIG_LOCAL"
    info "Update the paths in this file if your installations are in different locations."
}

# ---- Main ----
main() {
    echo "============================================"
    echo "   ASDMotion Local Setup"
    echo "============================================"
    echo ""

    check_prerequisites
    create_directories
    setup_asdmotion_env
    setup_mmaction_env
    setup_openpose
    download_models
    update_config

    echo ""
    echo "============================================"
    echo "   Setup Summary"
    echo "============================================"
    echo ""
    info "Two conda environments created:"
    echo "  - asdmotion   : Main pipeline (activate with: conda activate asdmotion)"
    echo "  - open-mmlab  : MMAction2/PoseC3D (used internally via run_in_env.sh)"
    echo ""
    info "Next steps:"
    echo "  1. Download model weights (see instructions above)"
    echo "  2. Install OpenPose (see instructions above)"
    echo "  3. Verify paths in: $RESOURCES_DIR/configs/config_local.yaml"
    echo "  4. Run ASDMotion:"
    echo ""
    echo "     # GUI mode:"
    echo "     conda activate asdmotion"
    echo "     python src/asdmotion/app/main_app.py"
    echo ""
    echo "     # Command-line mode:"
    echo "     conda activate asdmotion"
    echo "     python src/asdmotion/detector/executor.py \\"
    echo "       -cfg resources/configs/config_local.yaml \\"
    echo "       -video /path/to/your/video.mp4 \\"
    echo "       -out resources/runs"
    echo ""
    info "See LOCAL_SETUP.md for detailed instructions and troubleshooting."
}

main "$@"
