from os import path as osp

import numpy as np
from child_detector.detector import ChildDetector
from mmcv import Config

from jordi.logger import LogManager
from jordi.pipeline.openpose_executor import OpenposeInitializer
from jordi.pipeline.skeleton_layout import BODY_25_LAYOUT, COCO_LAYOUT
from jordi.pipeline.splitter import Splitter
from jordi.utils import read_pkl, write_pkl, get_video_properties, init_directories, create_config, save_config, RESOURCES_ROOT

CFG_DIR = osp.join(RESOURCES_ROOT, 'configs')
logger = LogManager.APP_LOGGER

class VideoTransformer:
    def __init__(self, work_dir, binary_model_name, openpose_root, detect_child, sequence_length, step_size, gpu_id, num_person_in, num_person_out):
        self.default_cfgs = {
            'binary': osp.join(CFG_DIR, 'binary_cfg_template.py'),
            'multiclass': osp.join(CFG_DIR, 'multiclass_cfg_template.py')
        }
        self.work_dir = osp.join(work_dir)
        self.gpu_id = gpu_id
        self.initializer = OpenposeInitializer(sequence_length=sequence_length, num_person_in=num_person_in, num_person_out=num_person_out,
                                               open_pose_path=openpose_root, as_img_dir=True, gpu_id=self.gpu_id)
        self.binary_model_name, self.detect_child, self.sequence_length, self.step_size = binary_model_name, detect_child, sequence_length, step_size
        if self.detect_child:
            self.child_detector = ChildDetector(device=self.gpu_id)

    def _create_skeleton(self, video_info):
        video_path = video_info['video_path']
        skeleton_path = video_info['skeleton_path']
        raw_path = video_info['raw_skeleton_path']
        if osp.exists(skeleton_path):
            logger.info(f'Skeleton already exists: {skeleton_path}')
            skeleton = read_pkl(skeleton_path)
        else:
            if osp.exists(raw_path):
                logger.info(f'Raw skeleton already exists: {raw_path}')
                skeleton = read_pkl(raw_path)
            else:
                logger.info(f'Initializing new skeleton: {skeleton_path}')
                skeleton_json = self.initializer.prepare_skeleton(video_path)
                skeleton = self.initializer.to_poseC3D(skeleton_json,
                                                       in_layout=BODY_25_LAYOUT, out_layout=COCO_LAYOUT)
                write_pkl(skeleton, raw_path)
            if self.detect_child:
                detections_path = video_info['detections_path']
                if osp.exists(detections_path):
                    logger.info(f'Detections already exists: {detections_path}')
                    detections = read_pkl(detections_path)
                else:
                    logger.info(f'Child detection in process: {video_path} , {skeleton_path}')
                    detections = self.child_detector.detect(video_path)
                    write_pkl(detections, detections_path)
                logger.info(f'Child detection - skeleton match in process: {video_path} , {skeleton_path}')
                skeleton = self.child_detector.match_skeleton(skeleton, detections, tolerance=200)
            else:
                T = video_info['properties']['frame_count']
                skeleton['child_ids'] = -np.ones(T)
                skeleton['child_detected'] = np.zeros(T)
                skeleton['child_bbox'] = np.zeros((T, 4))
        cids = skeleton['child_ids']
        valid_frames = cids[cids != -1].shape[0]
        last_valid_frame = int(np.where(cids != -1)[0][-1])
        video_info['properties']['valid_frames'] = valid_frames
        video_info['properties']['last_valid_frame'] = last_valid_frame
        write_pkl(skeleton, skeleton_path)
        return skeleton

    def prepare_dataset(self, video_info):
        basename = video_info['name']
        dataset_output = video_info['dataset_path']

        # if not osp.exists(dataset_output):
        logger.info(f'Creating new skeleton for {basename}')
        skeleton = self._create_skeleton(video_info)
        logger.info('Writing Dataset.')
        dataset = Splitter(skeleton, sequence_length=self.sequence_length, step_size=self.step_size, min_length=self.step_size*2).collect()
        out = {
            'split': {'test1': [f'{x["frame_dir"]}' for x in dataset]},
            'annotations': dataset
        }
        write_pkl(out, dataset_output)
        logger.info('Data initialized successfully.')

    def init_cfg(self, video_info, name, ann_file, model_type):
        logger.info(f'Initializing cfg for {name}')
        cfg = Config.fromfile(self.default_cfgs[model_type])
        if self.detect_child:
            for k in ['train', 'val', 'test']:
                cfg[f'{k}_pipeline'] = [{'type': 'ChildDetect'}] + cfg[f'{k}_pipeline']
        cfg['work_dir'] = video_info['jordi_dir']
        cfg['ann_file'] = ann_file
        cfg.data['train']['dataset']['pipeline'] = cfg[f'train_pipeline']
        for k in ['val', 'test']:
            cfg.data[k]['ann_file'] = cfg['ann_file']
            cfg.data[k]['pipeline'] = cfg[f'{k}_pipeline']
        cfg.dump(video_info[f'{model_type}_cfg_path'])

    def prepare_environment(self, video_path):
        fullname = osp.basename(video_path)
        name, ext = osp.splitext(fullname)
        work_dir = osp.join(self.work_dir, name)
        jordi_dir = osp.join(work_dir, 'jordi')
        model_dir = osp.join(jordi_dir, self.binary_model_name)
        resolution, fps, frame_count, length = get_video_properties(video_path)
        video_info = {
            'name': name,
            'fullname': fullname,
            'video_path': video_path,
            'work_dir': work_dir,
            'jordi_dir': jordi_dir,
            'skeleton_path': osp.join(jordi_dir, f'{name}.pkl'),
            'raw_skeleton_path': osp.join(jordi_dir, f'{name}_raw.pkl'),
            'dataset_path': osp.join(model_dir, f'{name}_dataset_{self.sequence_length}.pkl'),
            'binary_cfg_path':  osp.join(model_dir, f'{name}_binary_config.py'),
            'annotations_path': osp.join(model_dir, f'{name}_annotations.csv'),
            'conclusion_path': osp.join(model_dir, f'{name}_conclusion.csv'),
            'predictions_path': osp.join(model_dir, f'{name}_predictions.pkl'),
            'scores_path': osp.join(model_dir, f'{name}_scores.csv'),
            'self_path': osp.join(model_dir, f'{name}_exec_info.yaml'),
            'properties': {
                'resolution': resolution,
                'fps': fps,
                'frame_count': frame_count,
                'length': length
            }
        }
        if self.detect_child:
            video_info['child_detect'] = True
            video_info['detections_path'] = osp.join(work_dir, f'{name}_detections.pkl')
        init_directories(work_dir, jordi_dir, model_dir)
        video_info = create_config(video_info)
        self.init_cfg(video_info, name, video_info['dataset_path'], 'binary')
        self.prepare_dataset(video_info)
        save_config(video_info, video_info['self_path'])
        return video_info
