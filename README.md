# ASDPose

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Citation](#citation)


## Abstract
Stereotypical motor movements (SMMs) are a form of restricted and repetitive behaviors (RRBs) that are prevalent in individuals with Autism Spectrum Disorders (ASD). Previous studies attempting to quantify SMMs in ASD have relied on coarse and subjective reports or on manual annotation of video recordings. Here, we demonstrate the utility of a novel open-source AI algorithm that can analyze video recordings of children and automatically identify segments containing stereotypical movements.

## Requirements
1. OpenPose
2. PoseC3D (via MMAction2)

## Installation
### Prepare new environment:
> conda create -n asdpose python==3.9
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
[Follow the instructions]([https://github.com/open-mmlab/mmaction2](https://github.com/CMU-Perceptual-Computing-Lab/openpose))

### Install MMAction2:
[Follow the instructions](https://github.com/open-mmlab/mmaction2)

## Usage
> python detector.py -video ""
>
> -video
"//ac-s1/Data/Autism Center/recordings/Face camera - eye tracking experiment/control/707289343_2022-08-03-08-51-47 (2).mp4"
-out
"Z:\Users\TalBarami\models_outputs"
-gpu
1

## Citation
