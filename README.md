# ASDPose

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Citation](#citation)


## Abstract
Stereotypical motor movements (SMMs) are a form of restricted and repetitive behaviors (RRBs) that are prevalent in individuals with Autism Spectrum Disorders (ASD). Previous studies attempting to quantify SMMs in ASD have relied on coarse and subjective reports or on manual annotation of video recordings. Here, we demonstrate the utility of a novel open-source AI algorithm that can analyze video recordings of children and automatically identify segments containing stereotypical movements.

## Requirements
1. Python 3.9
2. OpenPose
3. PoseC3D (via MMAction2)

## Installation
### Prepare new environment:
```console
> conda create -n asdpose python==3.9
```
### Install required packages:
```console
> conda install esri::mmcv-full
> conda install conda-forge::ffmpeg
> conda install conda-forge::ffmpeg-python
> conda install conda-forge::omegaconf
> conda install anaconda::ipython
> conda install anaconda::seaborn
> conda install conda-forge::scikit-learn
```

### Install OpenPose:
[Follow the instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Install MMAction2:
[Follow the instructions](https://github.com/open-mmlab/mmaction2)

## Usage
```console
> python detector.py -cfg "<path_to_config_file>" -video "<path_to_video_file>" -out "<path_to_outputs_directory>"
```

### Configuration File:
```yaml
sequence_length: Length of each sequence to be predicted by PoseC3D. Default 200.
step_size: Step size of the sliding window that passes on the entire video. Default 30.
model_name: Name of the model inside resources/models directory Default 'jordi'.
classification_threshold: Threshold to classify an action as either SMM or not. Default 0.85.
child_detection: Will use YOLOv5 child detection module to detect the child per video frame. Default true .
num_person_in: Maximum number of people in each video frame. Default 5.
num_person_out: Maximum number of people in each skeleton sequence. Default 5.
open_pose_path: Path to openpose root directory.
mmaction_path: Path to mmaction2 root directory.
mmlab_python_path: Path to open-mmlab python executable.
```

### Outputs:
A directory with the name of the input video will be created. Inside it:
```yaml
jordi
|- jordi.pth
  |- <video_name>_annotations.csv - A table with start time, end time, movement type and stereotypical score of each segment.
  |- <video_name>_conclusion.csv - Summarize the annotations table with the sum of lengths of SMMs, the proportion of SMMs, the number of SMM segments, and the number of SMMs per minute.
  |- <video_name>_exec_info.yaml - Configuration file containing execution information.
  |- <video_name>_binary_config.py - Configuration file used to execute PoseC3D.
  |- <video_name>_predictions.pkl & <video_name>_scores.pkl - A per-sequence scores produced by PoseC3D for each sequence of <sequence_length> length while iterating over the entire video with step size <step_size>.
  |- <video_name>_dataset_<sequence_length>.pkl - Skeleton sequences that were fed to PoseC3D.
|- <video_name>raw.pkl - The skeleton sequence openpose produces.
|- <video_name>.pkl - The skeleton sequence after the matching process with the child detection module.
<video_name>_detections.pkl - Child detection outputs, as produced by the child detection module (Optional).
```

## Citation
