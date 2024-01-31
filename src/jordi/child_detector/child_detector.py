import logging
from os import path as osp

import torch
from torch.utils.data import DataLoader

from jordi.child_detector.iterable_video_dataset import IterableVideoDataset
from jordi.child_detector.skeleton_matcher import SkeletonMatcher
from jordi.utils import RESOURCES_ROOT


class ChildDetector:
    def __init__(self, batch_size=128, device=None):
        model_path = osp.join(RESOURCES_ROOT, 'models', 'child_detector.pt')
        handlers = list(logging.getLogger().handlers)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, _verbose=False)
        self.device = torch.device(device)
        if self.device is not None:
            self.model = self.model.to(self.device)
        logging.getLogger().handlers = handlers
        self.batch_size = batch_size

    def detect(self, video_path):
        dataset = IterableVideoDataset(video_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        dfs = []
        for frames_batch in dataloader:
            detections = self.model(frames_batch, size=640)
            dfs += detections.pandas().xywh
        return [(i, df) for i, df in enumerate(dfs)]

    def match_skeleton(self, skeleton, detections, iou_threshold=0.01, conf_threshold=0.1, similarity_threshold=0.85, grace_distance=20, tolerance=100):
        m = SkeletonMatcher(iou_threshold=iou_threshold, conf_threshold=conf_threshold, similarity_threshold=similarity_threshold, grace_distance=grace_distance, tolerance=tolerance)
        return m.match_skeleton(skeleton, detections)
