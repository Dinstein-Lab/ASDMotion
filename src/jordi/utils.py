import json
import pickle
from json import JSONDecodeError
from os import path as osp
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESOURCES_ROOT = osp.join(PROJECT_ROOT, 'resources')
EPSILON = 1e-4

def init_directories(*dirs):
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)


def create_config(dict_conf, out=None):
    for k, v in dict_conf.items():
        if type(v) == str and ('path' in k or 'dir' in k):
            dict_conf[k] = v.replace('\\', '/')
    config = OmegaConf.create(dict_conf)
    if out:
        with open(out.replace('\\', '/'), 'w') as fp:
            OmegaConf.save(config=config, f=fp.name)
    return config


def save_config(config, out):
    with open(out.replace('\\', '/'), 'w') as fp:
        OmegaConf.save(config=config, f=fp.name)


def load_config(file):
    with open(file.replace('\\', '/'), 'r') as fp:
        return OmegaConf.load(fp.name)


def get_video_properties(filename):
    try:
        vinf = ffmpeg.probe(filename)

        resolution_candidates = [(vinf['streams'][i]['width'], vinf['streams'][i]['height']) for i in range(len(vinf['streams'])) if 'width' in vinf['streams'][i].keys() and 'height' in vinf['streams'][i].keys()]
        fps_candidates = [vinf['streams'][i]['avg_frame_rate'] for i in range(len(vinf['streams'])) if 'avg_frame_rate' in vinf['streams'][i].keys()] + \
                         [vinf['streams'][i]['r_frame_rate'] for i in range(len(vinf['streams'])) if 'r_frame_rate' in vinf['streams'][i].keys()]
        fps_candidates = [x for x in fps_candidates if x != '0/0']

        resolution = resolution_candidates[0] if len(resolution_candidates) > 0 else None
        fps = eval(fps_candidates[0]) if len(fps_candidates) > 0 else None
        length_candidates = [vinf['streams'][i]['duration'] for i in range(len(vinf['streams'])) if 'duration' in vinf['streams'][i].keys()]
        if 'format' in vinf.keys() and 'duration' in vinf['format'].keys():
            length_candidates.append(vinf['format']['duration'])
        length = eval(length_candidates[0]) if len(length_candidates) > 0 else None
        if length is not None and fps is not None:
            estimated_frame = length * fps
        frame_candidates = [eval(vinf['streams'][i]['nb_frames']) for i in range(len(vinf['streams'])) if 'nb_frames' in vinf['streams'][i].keys()]
        frame_candidates = [f for f in frame_candidates if np.abs(f - estimated_frame) < np.min((50, estimated_frame * 0.1))]
        frame_count = int(np.max(frame_candidates)) if len(frame_candidates) > 0 else int(np.ceil(length * fps)) if length and fps else None
    except Exception:
        try:
            cap = cv2.VideoCapture(filename)
            resolution = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 6e5:
                frame_count = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
            length = frame_count / fps
        except Exception as e:
            raise e
        finally:
            cap.release()
    return resolution, fps, frame_count, length


def read_pkl(file):
    try:
        with open(file, 'rb') as p:
            return pickle.load(p)
    except (OSError, UnicodeDecodeError) as e:
        print(f'Error while reading {file}: {e}')
        raise e


def write_pkl(p, dst):
    with open(dst, 'wb') as f:
        pickle.dump(p, f)

def read_json(file):
    try:
        with open(file, 'rb') as j:
            return json.loads(j.read())
    except (OSError, UnicodeDecodeError, JSONDecodeError) as e:
        print(f'Error while reading {file}: {e}')
        raise e


def write_json(j, dst):
    with open(dst, 'w') as f:
        json.dump(j, f)

if __name__ == '__main__':
    cfg2 = load_config(osp.join(RESOURCES_ROOT, 'configs', 'config.yaml'))
    print()