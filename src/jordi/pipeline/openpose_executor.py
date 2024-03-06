import os
import shlex
import shutil
import subprocess
from enum import Enum
from itertools import chain
from os import path as osp

import cv2
import numpy as np
from tqdm import tqdm

from jordi.logger import LogManager
from jordi.pipeline.skeleton_layout import convert_layout, BODY_25_LAYOUT, COCO_LAYOUT
from jordi.utils import init_directories, get_video_properties, write_pkl, read_json

logger = LogManager.APP_LOGGER


class SkeletonSource(Enum):
    VIDEO = 'video'
    IMAGE = 'image_dir'
    WEBCAM = 'camera'


class OpenposeInitializer:
    def __init__(self, sequence_length, num_person_in, num_person_out, open_pose_path, as_img_dir=False, gpu_id=None):
        self.layout = BODY_25_LAYOUT
        self.C, self.T, self.V = 3, sequence_length, len(self.layout)
        self.num_person_in, self.num_person_out = num_person_in, num_person_out
        self.open_pose_path = open_pose_path
        self.as_img_dir = as_img_dir
        self.gpu_id = gpu_id

    def _video2img(self, video_path, out_path):
        name = osp.splitext(osp.basename(video_path))[0]
        logger.info(f'Converting video to image dir. Results will be written to: {out_path}')
        init_directories(out_path)
        cap = cv2.VideoCapture(video_path)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        d = len(str(n))
        i, ret = 0, True
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(osp.join(out_path, f'{name}_{str(i).zfill(d)}.jpg'), frame)
            i += 1
        cap.release()

    def _exec_openpose(self, src_path, skeleton_dst, source_type=SkeletonSource.VIDEO):
        init_directories(skeleton_dst)
        if src_path.startswith('\\\\'):
            src_path = f'\\{src_path}'
        params = {
            source_type.value: f'\"{src_path}\"',
            'model_pose': self.layout.model_pose,
            'write_json': f'\"{skeleton_dst}\"',
            'display': 0,
            'render_pose': 0
        }

        if self.layout.face:
            params['face'] = ''
            params['net_resolution'] = '256x256'
        if self.layout.hand:
            params['hand'] = ''
        if self.layout.name == 'BODY_21A':
            params['tracking'] = 1
        if self.gpu_id is not None:
            params['num_gpu'] = 1
            params['num_gpu_start'] = self.gpu_id

        args = ' '.join([f'--{k} {v}' for k, v in params.items()])

        cwd = os.getcwd()
        os.chdir(self.open_pose_path)
        cmd = f'build_windows/x64/Release/OpenPoseDemo.exe {args}' if osp.exists('build_windows') else f'bin/OpenPoseDemo.exe {args}'
        logger.info(f'Executing: {cmd}')
        try:
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
        finally:
            os.chdir(cwd)
            logger.info('OpenPose finished.')

    def prepare_skeleton(self, src_path, result_skeleton_dir=None, source_type=SkeletonSource.VIDEO, out_name=None):
        basename = osp.basename(src_path)
        basename_no_ext = osp.splitext(basename)[0] if source_type == SkeletonSource.VIDEO else basename

        process_dir = osp.join(self.open_pose_path, 'runs', basename_no_ext) if result_skeleton_dir is None else osp.join(result_skeleton_dir, basename_no_ext)
        openpose_output_path = osp.join(process_dir, 'openpose')

        try:
            resolution, fps, frame_count, length = get_video_properties(src_path)
            if self.as_img_dir:
                img_out_path = osp.join(process_dir, 'img_dirs')
                self._video2img(src_path, img_out_path)
                if frame_count is None or length is None:
                    frame_count = len(os.listdir(img_out_path))
                    length = frame_count / fps
                self._exec_openpose(img_out_path, openpose_output_path, source_type=SkeletonSource.IMAGE)
            else:
                self._exec_openpose(src_path, openpose_output_path, source_type=source_type)
            data = self.openpose_to_json(openpose_output_path)
            skeleton = {
                'name': basename,
                'video_path': src_path,
                'resolution': resolution,
                'fps': fps,
                'frame_count': frame_count,
                'length_seconds': length,
                'data': data,
            }
            if result_skeleton_dir:
                result_path = osp.join(result_skeleton_dir, out_name if out_name else f'{basename_no_ext}.json')
                write_pkl(skeleton, result_path)
            return skeleton
        except Exception as e:
            logger.error(f'Error creating skeleton from {src_path}: {e}')
            raise e
        finally:
            if osp.exists(process_dir):
                shutil.rmtree(process_dir)

    def openpose_to_json(self, openpose_dir):
        file_names = [osp.join(openpose_dir, f) for f in os.listdir(openpose_dir) if osp.isfile(osp.join(openpose_dir, f)) and f.endswith('json')]

        def collect_data(lst):
            k = np.array([float(c) for c in lst])
            x = np.round(k[::3], 8)
            y = np.round(k[1::3], 8)
            c = np.round(k[2::3], 8).tolist()
            return list(chain(*[(_x, _y) for (_x, _y) in zip(x, y)])), c

        result = []
        for i, file in tqdm(enumerate(file_names), ascii=True, desc='Openpose to JSON'):
            skeletons = []
            frame_info = read_json(file)
            people = frame_info['people']
            for pdx, p in enumerate(people):
                skeleton = {'person_id': p['person_id'] if p['person_id'] != [-1] else pdx}
                pose, score = collect_data(p['pose_keypoints_2d'])
                skeleton['pose'] = pose
                skeleton[f'pose_score'] = score
                skeletons.append(skeleton)
            result.append({'frame_index': i,
                           'skeleton': skeletons})
        return result

    def to_numpy(self, skeleton):
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for i, frame_info in tqdm(enumerate(skeleton['data']), ascii=True, desc='To numpy'):
            if i == self.T:
                break
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                pid = m if ('person_id' not in skeleton_info.keys()) else skeleton_info['person_id']
                if type(pid) == list:
                    pid = m if pid[0] < 0 else pid[0]
                pid %= self.num_person_in
                pose = skeleton_info['pose']
                score = skeleton_info['score'] if 'score' in skeleton_info.keys() else skeleton_info['pose_score']
                data_numpy[0, i, :, pid] = pose[0::2]
                data_numpy[1, i, :, pid] = pose[1::2]
                data_numpy[2, i, :, pid] = score

        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        return data_numpy[:, :, :, 0:self.num_person_out]

    def _to_posec3d_numpy(self, skeleton_data, in_layout, out_layout):
        keypoints = np.zeros((self.num_person_out, len(skeleton_data), len(out_layout), self.C - 1))
        scores = np.zeros((self.num_person_out, len(skeleton_data), len(out_layout)))

        for i, frame_info in enumerate(skeleton_data):
            skeletons = sorted(frame_info['skeleton'], key=lambda s: np.mean(s['pose_score']), reverse=True)[:self.num_person_out]
            for j, skeleton in enumerate(skeletons):
                keypoint, score = np.array([skeleton['pose'][::2], skeleton['pose'][1::2]]).T, np.array(skeleton['pose_score'])
                keypoints[j, i, :, :] = convert_layout(keypoint, in_layout, out_layout)
                scores[j, i, :] = convert_layout(score, in_layout, out_layout)
        return keypoints, scores

    def to_poseC3D(self, json_file, label=None, label_index=None, in_layout=BODY_25_LAYOUT, out_layout=COCO_LAYOUT):
        kp, s = self._to_posec3d_numpy(json_file['data'], in_layout, out_layout)

        result = {
            'keypoint': kp,
            'keypoint_score': s,
            'frame_dir': json_file['name'],
            'video_path': json_file['video_path'],
            'img_shape': json_file['resolution'],
            'original_shape': json_file['resolution'],
            'fps': json_file['fps'],
            'length_seconds': json_file['length_seconds'],
            'frame_count': json_file['frame_count'],
            'adjust': json_file['adjust'],
            'total_frames': len(json_file['data']),
        }
        if label is not None and label_index is not None:
            result['label_name'] = label
            result['label'] = label_index
        return result
