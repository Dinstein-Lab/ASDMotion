import shlex
import subprocess
from os import path as osp

import numpy as np
import pandas as pd

from asdpose.pipeline.aggregator import aggregate
from asdpose.logger import LogManager
from asdpose.utils import RESOURCES_ROOT, read_pkl

MODELS_DIR = osp.join(RESOURCES_ROOT, 'models')
logger = LogManager.APP_LOGGER


class Predictor:
    def __init__(self, work_dir, model_name, binary_threshold, labels, mmlab_python, mmaction_root, gpu_id=None):
        self.work_dir = work_dir
        self.model_name = model_name
        self.binary_model_path = osp.join(MODELS_DIR, self.model_name)
        self.threshold = binary_threshold

        self.labels = labels
        self.mmlab_python = mmlab_python
        self.mmaction_root = mmaction_root
        self.va_columns = ['video', 'video_full_name', 'video_path', 'start_time', 'end_time', 'start_frame', 'end_frame', 'movement', 'calc_date', 'annotator']
        self.gpu_id = gpu_id

    def _predict(self, cfg_path, model_path, out_path):
        if not osp.exists(out_path):
            out_exec = f'\\{out_path}' if out_path.startswith('\\\\') else out_path
            cmd = f'python "{osp.join(self.mmaction_root, "tools", "test.py")}" "{cfg_path}" "{model_path}" --out "{out_exec}"'
            if self.gpu_id is not None:
                cmd += f" --gpu-ids {self.gpu_id}"
            cmd = f'{osp.join(RESOURCES_ROOT, "run_in_env.bat")} {cmd}'.replace('\\', '/')
            logger.info(f'Executing: {cmd}')
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
            logger.info('Prediction complete successfully.')
        else:
            logger.info(f'Prediction exists: {out_path}')
        scores = read_pkl(out_path)
        return np.array(scores).T

    def _detect_stereotypical_movements(self, video_info):
        dataset = read_pkl(video_info['dataset_path'])['annotations']
        basename, fullname, path, fps = video_info['name'], video_info['fullname'], video_info['video_path'], video_info['properties']['fps']
        logger.info(f'Binary classification in progress')
        cfg_path, model_path, out_path = video_info['binary_cfg_path'], self.binary_model_path, video_info['predictions_path']
        binary_scores = self._predict(cfg_path, model_path, out_path)
        pos_score = binary_scores[1]

        df = pd.DataFrame(columns=self.va_columns + ['stereotypical_score'])
        for d, score in zip(dataset, pos_score):
            s, t = d['start'], d['end']
            df.loc[df.shape[0]] = [basename, fullname, path, s / fps, t / fps, s, t, -1, pd.Timestamp.now(), self.model_name, score]
        return df

    def _model_predictions(self, video_info):
        logger.info(f'Collecting ASDPose predictions for {video_info["name"]}')
        scores_path = video_info['scores_path']
        if osp.exists(scores_path):
            df = pd.read_csv(scores_path)
        else:
            df = self._detect_stereotypical_movements(video_info)
            df.to_csv(scores_path, index=False)
        agg = aggregate(df, self.threshold)
        agg['source'] = self.model_name
        return agg

    def conclude(self, _df, video_info):
        df = _df[_df['movement'] == 'Stereotypical'].copy()
        fps = video_info['properties']['fps']
        video_length_seconds = video_info['properties']['length']
        video_length_minute = video_length_seconds / 60
        video_frame_count = video_info['properties']['frame_count']
        valid_frames = video_info['properties']['valid_frames']
        last_valid_frame = video_info['properties']['last_valid_frame']
        df['segment_frames'] = df['end_frame'] - df['start_frame']
        df['relative_segment_frames'] = df['segment_frames'] / valid_frames
        df['segment_length_minute'] = df['segment_frames'] / (fps * 60)
        grp = df.groupby('video').agg({'segment_length_minute': 'sum', 'relative_segment_frames': 'sum', 'movement': 'count'}).reset_index()
        grp.columns = ['video', 'smm_length_minute', 'smm_proportion', 'smm_count']
        grp['fps'] = fps
        grp['video_length_minute'] = video_length_minute
        grp['video_frame_count'] = video_frame_count
        grp['valid_frames'] = valid_frames
        grp['last_valid_frame'] = last_valid_frame
        grp['smm/min'] = grp['smm_count'] / grp['video_length_minute']
        grp['assessment'] = grp['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
        return grp

    def annotate_video(self, video_info):
        df = self._model_predictions(video_info).sort_values(by=['video', 'start_time'])
        conc = self.conclude(df, video_info)
        df.to_csv(video_info['annotations_path'], index=False)
        conc.to_csv(video_info['conclusion_path'], index=False)
        return df
