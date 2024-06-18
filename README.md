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

### Downloads

[ASDMotion Dataset.](https://drive.google.com/file/d/1MiNIhlf4mL-vRW1ub2TP3nCYzfMW0bYt/view?usp=drive_link)

[Checkpoint weights for inference.](https://drive.google.com/file/d/1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE/view?usp=drive_link)

## Training
The entire training pipeline is managed through a forked repository of [MMAction2](https://github.com/TalBarami/mmaction2/tree/master/configs/skeleton/posec3d).
To train ASDPose, you need to follow the MMAction2 installation process. Once MMAction2 is installed, you can initiate the training by executing the following command:
```console
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example:
```console
python tools/train.py configs/skeleton/posec3d/asdmotion.py \
    --work-dir work_dirs/asdmotion \
    --validate --seed 0 --deterministic
```

For your convenience, we provide an example configuration file within the MMAction2 repository, which can be found at `/configs/skeleton/posec3d/asdmotion.py`. 
This configuration file is tailored specifically for training ASDPose and includes all the necessary parameters and settings to get you started efficiently.

## Inference
You can test the model on a pre-defined skeleton dataset using the [MMAction2](https://github.com/TalBarami/mmaction2/tree/master/configs/skeleton/posec3d) repository. To do this, execute the following command:

```console
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

In addition to testing, we provide tools for extracting a skeletal representation from a video and creating a dataset file. This dataset corresponds to a sliding window that iterates over the video and classifies each segment as either containing a stereotypical motor movement (SMM) or not.

We offer both graphical user interface (GUI) and command-line interface (CMD) applications for this task.

To use the GUI, execute:
```console
> python src/asdmotion/app/main_app.py
```
Ensure that the configuration file located at `/resources/config/config.yaml` contains the correct paths to OpenPose and MMAction2.

To use the CMD application, execute:
```console
> python src/asdmotion/detector/detector.py -cfg "<path_to_config_file>" -video "<path_to_video_file>" -out "<path_to_outputs_directory>"
```

### Configuration File:
Each execution of ASDMotion relies on a set of customizable configurations, which can be specified as follows:

```yaml
sequence_length: Length of each sequence to be predicted by PoseC3D. Default is 200.
step_size: Step size of the sliding window that processes the entire video. Default is 30.
model_name: Name of the model inside the resources/models directory. Default is 'asdmotion'.
classification_threshold: Threshold to classify an action as either SMM or not. Default is 0.85.
child_detection: Utilizes the YOLOv5 child detection module to detect the child in each video frame. Default is true.
num_person_in: Maximum number of people in each video frame. Default is 5.
num_person_out: Maximum number of people in each skeleton sequence. Default is 5.
open_pose_path: Path to the OpenPose root directory.
mmaction_path: Path to the MMAction2 root directory.
mmlab_python_path: Path to the OpenMMLab Python executable.
```

If you are using this repository for the first time, ensure to update the configuration file with the appropriate paths and settings.

### Outputs:

Upon execution, a directory named after the input video will be created. Inside this directory, you will find the following structure:

```yaml
├── asdmotion
│   ├── asdmotion.pth
│   │   ├──  <video_name>_annotations.csv - A table with start time, end time, movement type, and stereotypical score of each segment.
│   │   ├──  <video_name>_conclusion.csv - Summarizes the annotations table with the total length of SMMs, the proportion of SMMs, the number of SMM segments, and the number of SMMs per minute.
│   │   ├──  <video_name>_exec_info.yaml - Configuration file containing execution information.
│   │   ├──  <video_name>_binary_config.py - Configuration file used to execute PoseC3D.
│   │   ├──  <video_name>_predictions.pkl & <video_name>_scores.pkl - Per-sequence scores produced by PoseC3D for each sequence of <sequence_length> length while iterating over the entire video with step size <step_size>.
│   │   └──  <video_name>_dataset_<sequence_length>.pkl - Skeleton sequences that were fed to PoseC3D.
│   ├── <video_name>raw.pkl - The skeleton sequence produced by OpenPose.
│   └── <video_name>.pkl - The skeleton sequence after the matching process with the child detection module.
└── <video_name>_detections.pkl - Child detection outputs produced by the child detection module (optional).
```

An example of a video segment where SMM is observed, along with the signal produced by the model:
<p align="center">
  <img src="/resources/sample.gif" alt="Example" width="500"/>
</p>

## Citation
If you find this project useful in your research, please consider citing:
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
