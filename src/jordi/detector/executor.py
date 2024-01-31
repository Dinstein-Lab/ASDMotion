import time
from argparse import ArgumentParser
from os import path as osp

from omegaconf import OmegaConf

from jordi.detector.detector import Predictor
from jordi.detector.preprocess import VideoTransformer
from jordi.logger import LogManager
from jordi.utils import load_config

logger = LogManager.APP_LOGGER

def predict_video(vt, vpath, p=None):
    v = osp.splitext(osp.basename(vpath))[0]
    logger.info(f'Starting video creation: {vpath}\n\tResults will be saved to {osp.join(vt.work_dir, v)}')
    s = time.time()
    vinf = vt.prepare_environment(vpath)
    if p is not None:
        p.annotate_video(vinf)
    t = time.time()
    delta = t - s
    logger.info(f'Total {int(delta // 3600):02d}:{int((delta % 3600) // 60):02d}:{delta % 60:05.2f} for {v}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str)
    parser.add_argument("-video", "--video_path")
    parser.add_argument("-out", "--out_path")
    parser.add_argument("-gpu", "--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.merge(load_config(args.cfg), OmegaConf.create(vars(args)))

    video_path = cfg.video_path
    if not osp.exists(video_path):
        raise FileNotFoundError(f'Video path {video_path} does not exist.')
    work_dir = cfg.out_path

    logger.info(f'Executing JORDI on {video_path}. Results will be saved to {work_dir}')
    vt = VideoTransformer(work_dir, cfg.model_name, cfg.open_pose_path, cfg.child_detection, cfg.sequence_length, cfg.step_size, cfg.gpu, cfg.num_person_in, cfg.num_person_out)
    p = Predictor(work_dir, cfg.model_name, cfg.classification_threshold, ['NoAction', 'Stereotypical'], cfg.mmlab_python_path, cfg.mmaction_path, cfg.gpu)
    logger.info(f'Annotating: {video_path}')
    predict_video(vt=vt, p=p, vpath=video_path)
