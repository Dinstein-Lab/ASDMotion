# ASDMotion on MacBook Pro M1 - Setup Guide

Your M1 Mac has **no NVIDIA GPU**, so CUDA is not available. This guide uses **Docker** to run everything in a Linux container with CPU mode. It works reliably but processing is slower than on a GPU machine.

**Expected processing time**: ~1-2 hours per 5 minutes of video (CPU mode).

Tip: Record shorter clips (3-5 minutes) to keep processing time manageable.

---

## Prerequisites

### 1. Install Docker Desktop for Mac

Download and install Docker Desktop (Apple Silicon version):
https://www.docker.com/products/docker-desktop/

After installing:
- Open Docker Desktop
- Go to Settings > Resources
- Set Memory to **at least 8 GB** (10+ GB recommended)
- Set CPU to **at least 4 cores**
- Click "Apply & Restart"

### 2. Download Model Weights

Download the pre-trained ASDMotion model:

**Direct link**: https://drive.google.com/file/d/1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE/view

Save the downloaded file as `asdmotion.pth` into the `resources/models/` folder of this project.

If you have Python with `gdown` installed, you can also do:
```bash
pip install gdown
gdown "1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE" -O resources/models/asdmotion.pth
```

---

## Setup (One-Time)

Open Terminal and navigate to this project:

```bash
cd /path/to/ASDMotion
```

Build the Docker image (this takes 20-40 minutes the first time):

```bash
docker build --platform linux/amd64 -t asdmotion .
```

> The `--platform linux/amd64` flag is important on M1. Docker will use Rosetta 2
> to emulate x86, which is necessary because OpenPose and mmcv don't have ARM builds.

Create folders for your videos and output:

```bash
mkdir -p videos output
```

---

## Running ASDMotion

### Step 1: Place Your Video

Copy the video of your child into the `videos/` folder:

```bash
cp /path/to/my_video.mp4 videos/
```

Supported formats: `.mp4`, `.avi`

### Step 2: Run the Analysis

```bash
docker run --platform linux/amd64 \
  -v "$(pwd)/videos:/data/input:ro" \
  -v "$(pwd)/output:/data/output" \
  -v "$(pwd)/resources/models:/app/resources/models" \
  asdmotion /data/input/my_video.mp4
```

Replace `my_video.mp4` with your actual video filename.

Or using docker compose:

```bash
docker compose run asdmotion /data/input/my_video.mp4
```

### Step 3: Wait for Processing

The terminal will show progress. On an M1 Mac with CPU mode:
- **3-minute video**: ~30-60 minutes
- **5-minute video**: ~1-2 hours
- **10-minute video**: ~2-4 hours

You can let it run in the background. Don't close the terminal.

### Step 4: View Results

Results are saved in `output/<video_name>/asdmotion/asdmotion.pth/`:

```bash
# View the summary (key metrics)
cat output/my_video/asdmotion/asdmotion.pth/my_video_conclusion.csv

# View detailed annotations (each detected segment)
cat output/my_video/asdmotion/asdmotion.pth/my_video_annotations.csv
```

You can open the CSV files in any spreadsheet app (Excel, Numbers, Google Sheets).

---

## Understanding the Results

### Conclusion CSV (Summary)

| Column | Meaning |
|---|---|
| `smm_length_minute` | Total duration of stereotypical movements (minutes) |
| `smm_proportion` | What fraction of the video contains SMMs (0.0 to 1.0) |
| `smm_count` | Number of separate SMM episodes detected |
| `smm/min` | Rate of stereotypical movements per minute |

### Annotations CSV (Details)

| Column | Meaning |
|---|---|
| `start_time` / `end_time` | When each segment starts/ends (seconds) |
| `movement` | `Stereotypical` or `NoAction` |
| `stereotypical_score` | Model confidence (0.0 to 1.0, threshold default: 0.85) |

---

## Tips for Recording Videos of Your Child

For best detection results:

1. **Full body visible** - Place the camera so your child's entire body is in frame
2. **Stable camera** - Use a tripod or place the phone/camera on a stable surface
3. **Good lighting** - Natural daylight or well-lit room, avoid backlighting
4. **Plain background** - A simple background helps the pose estimation
5. **Single subject** - If possible, record just your child (no other people in frame).
   If others are present, the child detector can help, but it works best with just the child
6. **Natural setting** - Let your child behave naturally in a familiar space
7. **Short clips** - 3-5 minute clips are ideal for processing time
8. **MP4 format** - Most phone cameras record in MP4, which works well

### What the Tool Detects

The model was trained to identify stereotypical motor movements including:
- Hand/arm flapping
- Body rocking
- Repetitive hand movements
- Other restricted/repetitive motor behaviors

---

## Adjusting Sensitivity

If you want to detect more subtle movements (at the risk of more false positives),
lower the classification threshold in the config:

```bash
# Create a custom config
cp resources/configs/config_docker.yaml resources/configs/my_config.yaml
```

Edit `my_config.yaml` and change `classification_threshold` from `0.85` to e.g. `0.7`.

Then run with your custom config (mount it into the container):

```bash
docker run --platform linux/amd64 \
  -v "$(pwd)/videos:/data/input:ro" \
  -v "$(pwd)/output:/data/output" \
  -v "$(pwd)/resources/models:/app/resources/models" \
  -v "$(pwd)/resources/configs/my_config.yaml:/app/resources/configs/config_docker.yaml:ro" \
  asdmotion /data/input/my_video.mp4
```

---

## Disabling Child Detection

By default in the Docker config, child detection is **disabled** (`child_detection: false`)
to simplify the setup. This works well when only the child is visible in the video.

If your video has multiple people and you need child detection:

1. Download the child detector model to `resources/models/child_detector.pt`
2. Set `child_detection: true` in your config

---

## Troubleshooting

### "Model weights not found"
Make sure `resources/models/asdmotion.pth` exists. Download it from the Google Drive link above.

### Docker build fails
- Ensure Docker Desktop is running
- Ensure you have at least 15 GB free disk space
- Try: `docker build --platform linux/amd64 --no-cache -t asdmotion .`

### Out of memory
- Open Docker Desktop > Settings > Resources
- Increase Memory to 10 GB or more
- Restart Docker Desktop

### "No children detected in video"
- This happens when the pose estimation can't find a person in the frames
- Try a video with better lighting and the child's full body clearly visible
- Make sure the child is moving in the video (not sitting perfectly still)

### Very slow processing
This is expected on M1 with CPU mode. Suggestions:
- Use shorter video clips (3-5 minutes)
- Close other heavy apps while processing
- Let it run overnight for longer videos

### Container exits immediately
Run interactively to see errors:
```bash
docker run --platform linux/amd64 -it \
  -v "$(pwd)/videos:/data/input:ro" \
  -v "$(pwd)/output:/data/output" \
  -v "$(pwd)/resources/models:/app/resources/models" \
  asdmotion bash
```
Then manually run:
```bash
python src/asdmotion/detector/executor.py \
  -cfg resources/configs/config_docker.yaml \
  -video /data/input/my_video.mp4 \
  -out /data/output
```

---

## Important Disclaimer

This is a **research tool** published in a peer-reviewed study (JAMA Network Open, 2024).
- It detects patterns in movement; it is **not** a diagnostic tool
- Results should be shared with and interpreted by qualified clinicians
- The tool supplements but does not replace professional assessment
- No medical decisions should be made based solely on these results
