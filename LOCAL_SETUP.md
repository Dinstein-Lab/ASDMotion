# ASDMotion - Local Setup Guide

This guide walks you through setting up ASDMotion on your local machine to analyze videos for stereotypical motor movements (SMMs) in children with ASD.

## What This Tool Does

ASDMotion takes a video recording of a child and:
1. Extracts skeletal pose data from each frame (via OpenPose)
2. Optionally detects which person in the frame is the child (via YOLOv5)
3. Classifies video segments as containing stereotypical movements or not (via PoseC3D)
4. Outputs CSV files with timestamps, movement types, and quantitative metrics

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10, Ubuntu 18.04+, macOS | Ubuntu 20.04+ or Windows 10/11 |
| Python | 3.9 | 3.9 (exact) |
| RAM | 8 GB | 16 GB+ |
| GPU | None (CPU works, slow) | NVIDIA GPU with CUDA 11.7+ |
| Disk | 5 GB | 10 GB+ |
| Software | Conda, ffmpeg, git | Conda, ffmpeg, git, CMake (for OpenPose) |

---

## Quick Start (Automated)

Run the setup script to install everything:

```bash
chmod +x setup_local.sh
./setup_local.sh
```

The script will:
- Create conda environments (`asdmotion` and `open-mmlab`)
- Install all Python dependencies
- Clone MMAction2
- Guide you through OpenPose installation and model downloads

---

## Manual Step-by-Step Setup

### Step 1: Install Prerequisites

```bash
# Install Miniconda (if not installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Install ffmpeg
sudo apt update && sudo apt install -y ffmpeg git cmake
```

### Step 2: Create the Main Environment

```bash
conda create -n asdmotion python=3.9 -y
conda activate asdmotion
pip install -r requirements_local.txt
```

### Step 3: Install mmcv (required dependency)

```bash
# With CUDA (recommended):
pip install openmim
mim install mmcv-full==1.7.1

# Without CUDA (CPU only):
pip install mmcv
```

### Step 4: Install Optional Components

```bash
# Child detector (recommended for videos with multiple people)
pip install git+https://github.com/TalBarami/Child-Detector.git

# Skeleton tools
pip install git+https://github.com/TalBarami/SkeletonTools.git
```

### Step 5: Set Up MMAction2 (Separate Environment)

MMAction2 runs in its own environment because it may have conflicting dependencies:

```bash
# Create environment
conda create -n open-mmlab python=3.9 -y
conda activate open-mmlab

# Install PyTorch (with CUDA)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Or PyTorch CPU-only:
# pip install torch==1.13.1+cpu torchvision==0.14.1+cpu \
#     -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install mmcv and mmaction2
pip install openmim
mim install mmcv-full==1.7.1

git clone https://github.com/TalBarami/mmaction2.git deps/mmaction2
cd deps/mmaction2
pip install -e .
cd ../..

conda deactivate
```

### Step 6: Install OpenPose

OpenPose is the most complex dependency. Choose one option:

#### Option A: Build from Source (Linux)

```bash
# Install build dependencies
sudo apt install -y cmake libopencv-dev protobuf-compiler libprotobuf-dev

git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git deps/openpose
cd deps/openpose
git submodule update --init --recursive

mkdir build && cd build

# With GPU:
cmake .. -DBUILD_PYTHON=OFF -DDOWNLOAD_BODY_25_MODEL=ON -DGPU_MODE=CUDA

# Without GPU (CPU only - slower but works):
# cmake .. -DBUILD_PYTHON=OFF -DDOWNLOAD_BODY_25_MODEL=ON -DGPU_MODE=CPU_ONLY

make -j$(nproc)
cd ../../..
```

#### Option B: Pre-built (Windows)

Download the portable version from:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases

Extract to `deps/openpose/`.

#### Option C: Docker (Linux - Easiest)

```bash
# Pull OpenPose Docker image
docker pull cwaffles/openpose

# Run OpenPose on a video:
docker run --gpus all -v /path/to/videos:/data cwaffles/openpose \
    --video /data/input.mp4 --write_json /data/output/ --display 0 --render_pose 0
```

### Step 7: Download Model Weights

Download the pre-trained model checkpoint:

1. **ASDMotion checkpoint** (REQUIRED):
   - URL: https://drive.google.com/file/d/1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE/view
   - Save to: `resources/models/asdmotion.pth`

2. **Child Detector model** (auto-downloaded by YOLOv5, or manual):
   - Save to: `resources/models/child_detector.pt`

```bash
# Create the models directory
mkdir -p resources/models

# If you have gdown installed:
pip install gdown
gdown "1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE" -O resources/models/asdmotion.pth
```

### Step 8: Configure Paths

Create a local configuration file:

```bash
cp resources/configs/config.yaml resources/configs/config_local.yaml
```

Edit `resources/configs/config_local.yaml` with your actual paths:

```yaml
sequence_length: 200
step_size: 30
model_name: 'asdmotion'
child_detection: true
classification_threshold: 0.85
num_person_in: 5
num_person_out: 5
open_pose_path: '/full/path/to/deps/openpose'
mmaction_path: '/full/path/to/deps/mmaction2'
mmlab_python_path: '/full/path/to/miniconda3/envs/open-mmlab/bin/python'
```

To find your `mmlab_python_path`:
```bash
conda activate open-mmlab
which python
conda deactivate
```

### Step 9: Update run_in_env.sh (Linux only)

Edit `resources/run_in_env.sh` to match your conda installation path:

```bash
CONDA_PATH="$HOME/miniconda3"   # or $HOME/anaconda3
ENV_NAME="open-mmlab"
```

---

## Running ASDMotion

### GUI Mode

```bash
conda activate asdmotion
python src/asdmotion/app/main_app.py
```

The GUI lets you:
- Browse and select video files (.mp4, .avi)
- Toggle child detection on/off
- Toggle ASDMotion analysis on/off
- Click "Start" to process

### Command-Line Mode

```bash
conda activate asdmotion
python src/asdmotion/detector/executor.py \
    -cfg "resources/configs/config_local.yaml" \
    -video "/path/to/your/video.mp4" \
    -out "resources/runs"
```

### Understanding the Output

After processing, results are saved in `resources/runs/<video_name>/`:

| File | Description |
|---|---|
| `*_annotations.csv` | Each row is a video segment with start/end times, movement type, and stereotypical score |
| `*_conclusion.csv` | Summary: total SMM duration, proportion, count, and SMMs per minute |
| `*_exec_info.yaml` | Configuration used for this run |

Key metrics in the conclusion:
- **smm_length_minute**: Total duration of stereotypical movements (minutes)
- **smm_proportion**: Fraction of video containing SMMs
- **smm_count**: Number of separate SMM episodes
- **smm/min**: Rate of stereotypical movements per minute

---

## Tips for Recording Videos

For best results when recording your child:

1. **Camera placement**: Place the camera at a fixed position where the child's full body is visible
2. **Lighting**: Ensure good, even lighting (avoid backlighting)
3. **Background**: A plain, uncluttered background works best
4. **Duration**: 5-15 minute recordings work well
5. **Resolution**: 720p or higher
6. **Format**: MP4 or AVI
7. **Single subject**: If possible, record the child alone. If others are present, enable child detection
8. **Natural behavior**: Let the child behave naturally in a familiar setting

---

## Troubleshooting

### "No children detected in video"
- Try setting `child_detection: false` in your config if the video has only the child
- Ensure the child's full body is visible in the frame

### OpenPose errors
- Verify the `open_pose_path` in your config points to the OpenPose root directory
- On Linux, the binary should be at `<openpose_root>/build/examples/openpose/openpose.bin`
- On Windows, it should be at `<openpose_root>/build_windows/x64/Release/OpenPoseDemo.exe`

### CUDA / GPU errors
- Check your CUDA version: `nvidia-smi`
- Ensure PyTorch was installed with matching CUDA version
- For CPU-only: install PyTorch CPU version and set `gpu_ids=[]` in the config template

### MMAction2 errors
- Verify `mmaction_path` points to the cloned mmaction2 directory
- Verify `mmlab_python_path` points to the Python binary inside the `open-mmlab` conda env
- On Linux, update `resources/run_in_env.sh` with correct conda path

### Import errors for mmcv
- The `preprocess.py` file imports `from mmcv import Config`
- Install mmcv: `pip install mmcv` or `mim install mmcv-full`

### Permission errors on Linux
```bash
chmod +x resources/run_in_env.sh
chmod +x setup_local.sh
```

---

## Important Notes

- This is a **research tool** based on a peer-reviewed study published in JAMA Network Open
- Results should be interpreted by qualified professionals
- The tool identifies patterns in movement; it is not a diagnostic tool
- The default classification threshold (0.85) is conservative; lower it to detect more subtle movements, raise it to reduce false positives
- Processing time depends on video length and hardware (GPU is strongly recommended)
