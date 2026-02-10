# =============================================================================
# ASDMotion Dockerfile
# =============================================================================
# Bundles OpenPose + MMAction2 + ASDMotion into a single container.
# Works on any platform including Apple Silicon M1/M2/M3 (CPU mode).
#
# Build:  docker build -t asdmotion .
# Run:    docker run -v /path/to/videos:/data/input -v /path/to/output:/data/output asdmotion /data/input/video.mp4
# =============================================================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- System dependencies ----
# Ubuntu 22.04 ships with Python 3.10; using it directly (ASDMotion works with 3.9+)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libboost-all-dev \
    libhdf5-dev \
    libatlas-base-dev \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python' command is available
RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# ---- Build OpenPose (CPU mode) ----
WORKDIR /opt
RUN git clone --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git && \
    cd openpose && \
    git submodule update --init --recursive

WORKDIR /opt/openpose/build
RUN cmake .. \
    -DBUILD_PYTHON=OFF \
    -DDOWNLOAD_BODY_25_MODEL=ON \
    -DDOWNLOAD_BODY_COCO_MODEL=OFF \
    -DDOWNLOAD_BODY_MPI_MODEL=OFF \
    -DDOWNLOAD_FACE_MODEL=OFF \
    -DDOWNLOAD_HAND_MODEL=OFF \
    -DGPU_MODE=CPU_ONLY \
    -DUSE_MKL=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    && make -j$(nproc)

# ---- Install PyTorch (CPU) ----
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# ---- Install mmcv and MMAction2 ----
RUN pip install openmim && \
    mim install mmcv-full==1.7.1 || pip install mmcv

WORKDIR /opt
RUN git clone --depth 1 https://github.com/TalBarami/mmaction2.git && \
    cd mmaction2 && \
    pip install -e .

# ---- Install ASDMotion dependencies ----
COPY requirements_local.txt /tmp/requirements_local.txt
RUN pip install -r /tmp/requirements_local.txt && rm /tmp/requirements_local.txt

# Install optional components
RUN pip install git+https://github.com/TalBarami/Child-Detector.git || true
RUN pip install git+https://github.com/TalBarami/SkeletonTools.git || true
RUN pip install gdown

# ---- Copy ASDMotion source ----
WORKDIR /app
COPY . /app

# Create required directories
RUN mkdir -p /app/resources/models /app/resources/logs /app/resources/runs /data/input /data/output

# ---- Configure paths ----
RUN cat > /app/resources/configs/config_docker.yaml <<EOF
sequence_length: 200
step_size: 30
model_name: 'asdmotion'
child_detection: false
classification_threshold: 0.85
num_person_in: 5
num_person_out: 5
open_pose_path: '/opt/openpose'
mmaction_path: '/opt/mmaction2'
mmlab_python_path: '/usr/bin/python'
EOF

# ---- Create entrypoint script ----
RUN cat > /app/entrypoint.sh <<'ENTRY'
#!/bin/bash
set -e

MODEL_PATH="/app/resources/models/asdmotion.pth"

# Check if model weights exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "============================================"
    echo "  Model weights not found!"
    echo "============================================"
    echo ""
    echo "You need to download the ASDMotion model checkpoint."
    echo ""
    echo "Option 1: Download manually and mount:"
    echo "  Download from: https://drive.google.com/file/d/1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE/view"
    echo "  Then run with: docker run -v /path/to/asdmotion.pth:/app/resources/models/asdmotion.pth ..."
    echo ""
    echo "Option 2: Auto-download (attempting now)..."
    gdown "1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE" -O "$MODEL_PATH" 2>/dev/null || {
        echo "Auto-download failed. Please download manually."
        exit 1
    }
    echo "Model downloaded successfully!"
fi

# If a video path argument is given, run analysis
if [ $# -gt 0 ]; then
    VIDEO_PATH="$1"
    OUTPUT_DIR="${2:-/data/output}"

    if [ ! -f "$VIDEO_PATH" ]; then
        echo "Error: Video file not found: $VIDEO_PATH"
        echo "Usage: docker run -v /your/videos:/data/input -v /your/output:/data/output asdmotion /data/input/video.mp4"
        exit 1
    fi

    echo "============================================"
    echo "  ASDMotion Analysis"
    echo "============================================"
    echo "  Video:  $VIDEO_PATH"
    echo "  Output: $OUTPUT_DIR"
    echo "============================================"
    echo ""
    echo "NOTE: Running on CPU. Processing may take several hours for long videos."
    echo "      A 5-minute video takes approximately 1-2 hours on CPU."
    echo ""

    cd /app
    python src/asdmotion/detector/executor.py \
        -cfg "resources/configs/config_docker.yaml" \
        -video "$VIDEO_PATH" \
        -out "$OUTPUT_DIR"

    echo ""
    echo "============================================"
    echo "  Analysis Complete!"
    echo "============================================"
    echo "  Results saved to: $OUTPUT_DIR"
    echo "  Check the *_conclusion.csv for summary metrics."
    echo "============================================"
else
    echo "Usage: docker run -v /your/videos:/data/input -v /your/output:/data/output asdmotion /data/input/video.mp4"
    echo ""
    echo "Or start interactive shell: docker run -it asdmotion bash"
fi
ENTRY
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
