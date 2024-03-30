<p align="center">
  <img src="/resources/logo_jordi.webp" alt="ASDMotion" width="500"/>
</p>


# ASDMotion

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Citation](#citation)


## Abstract
This repository holds the codebase, dataset and models for the paper:

**Automated identification and quantification of stereotypical movements from video recordings of children with ASD**, Barami Tal, Manelis-Baram Liora, Kaiser Hadas, Dinstein Ilan; 2024. [Arxiv Preprint.](https://www.biorxiv.org/content/early/2024/03/06/2024.03.02.582828)

Stereotypical motor movements (SMMs) are a form of restricted and repetitive behaviors (RRBs) that are prevalent in individuals with Autism Spectrum Disorders (ASD). Previous studies attempting to quantify SMMs in ASD have relied on coarse and subjective reports or on manual annotation of video recordings. Here, we demonstrate the utility of a novel open-source AI algorithm that can analyze video recordings of children and automatically identify segments containing stereotypical movements.


## Requirements
This code was tested using:
1. Windows 10/11
2. Python 3.9
3. Pytorch 1.13
4. CUDA 11.7

Other OS/Python distributions are expected to work.

## Installation
### Prepare new environment:
```console
> conda create -n asdmotion python==3.9
```
### Install required packages:
```console
> pip install -r requirements.txt
```

### Install OpenPose:
This repository utilizes OpenPose to extract the skeletal representation of individuals per video frame.
The OpenPose demo version is sufficient for this task. For installation, [follow the instructions.](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md)

### Install MMAction2:
The SMM identification model is based on PoseC3D, a skeleton-based action recognition model. 
For our setup, it is sufficient to clone the repository and install its dependencies on a **separate** environment.
[Follow the instructions (Forked repository)](https://github.com/TalBarami/mmaction2)


### Install the child detector (Optional)
The child detector allows more accurate detections when the video contains multiple adults and one child.
```console
> git clone https://github.com/TalBarami/Child-Detector.git
> cd Child-Detector
> python setup.py develop
```

## Usage
We provide both GUI and command-line applications.
For GUI, execute:
```console
> python src/jordi/app/main_app.py
```
Ensure the configuration file `/resources/config/config.yaml` contains the paths to OpenPose and MMAction2.

For CMD, execute:
```console
> python src/jordi/detector/detector.py -cfg "<path_to_config_file>" -video "<path_to_video_file>" -out "<path_to_outputs_directory>"
```

### Configuration File:
Each execution of ASDMotion depends on a set of customizable configurations:
```yaml
sequence_length: Length of each sequence to be predicted by PoseC3D. Default 200.
step_size: Step size of the sliding window that passes on the entire video. Default 30.
model_name: Name of the model inside resources/models directory Default 'asdmotion'.
classification_threshold: Threshold to classify an action as either SMM or not. Default 0.85.
child_detection: Will use YOLOv5 child detection module to detect the child per video frame. Default true .
num_person_in: Maximum number of people in each video frame. Default 5.
num_person_out: Maximum number of people in each skeleton sequence. Default 5.
open_pose_path: Path to openpose root directory.
mmaction_path: Path to mmaction2 root directory.
mmlab_python_path: Path to open-mmlab python executable.
```
If you use this repository for the first time, make sure to update the 

### Outputs:
A directory with the name of the input video will be created. Inside it:
```yaml
├── asdmotion
│   ├── asdmotion.pth
│   │   ├──  <video_name>_annotations.csv - A table with start time, end time, movement type and stereotypical score of each segment.
│   │   ├──  <video_name>_conclusion.csv - Summarize the annotations table with the sum of lengths of SMMs, the proportion of SMMs, the number of SMM segments, and the number of SMMs per minute.
│   │   ├──  <video_name>_exec_info.yaml - Configuration file containing execution information.
│   │   ├──  <video_name>_binary_config.py - Configuration file used to execute PoseC3D.
│   │   ├──  <video_name>_predictions.pkl & <video_name>_scores.pkl - A per-sequence scores produced by PoseC3D for each sequence of <sequence_length> length while iterating over the entire video with step size <step_size>.
│   │   └──  <video_name>_dataset_<sequence_length>.pkl - Skeleton sequences that were fed to PoseC3D.
│   ├── <video_name>raw.pkl - The skeleton sequence openpose produces.
│   └── <video_name>.pkl - The skeleton sequence after the matching process with the child detection module.
└── <video_name>_detections.pkl - Child detection outputs, produced by the child detection module (Optional).
```

An example of video segment where SMM is observed, along with the signal produced by the model:
<p align="center">
  <img src="/resources/sample.gif" alt="Example" width="500"/>
</p>

## Citation
If you find this project useful in your research, please consider cite:
```BibTeX
@article {Barami2024.03.02.582828,
	author = {Tal Barami and Liora Manelis-Baram and Hadas Kaiser and Michal Ilan and Aviv Slobodkin and Ofri Hadashi and Dor Hadad and Danel Waissengreen and Tanya Nitzan and Idan Menashe and Analya Michaelovsky and Michal Begin and Ditza A. Zachor and Yair Sadaka and Judah Koler and Dikla Zagdon and Gal Meiri and Omri Azencot and Andrei Sharf and Ilan Dinstein},
	title = {Automated identification and quantification of stereotypical movements from video recordings of children with ASD},
	elocation-id = {2024.03.02.582828},
	year = {2024},
	doi = {10.1101/2024.03.02.582828},
	URL = {https://www.biorxiv.org/content/early/2024/03/06/2024.03.02.582828},
	eprint = {https://www.biorxiv.org/content/early/2024/03/06/2024.03.02.582828.full.pdf},
	journal = {bioRxiv}
}
```
