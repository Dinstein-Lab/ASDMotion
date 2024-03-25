import numpy as np
from tqdm import tqdm

from asdmotion.child_detector.utils import bounding_box, get_box, get_iou, find_nearest


def get_boxes(kp, score):
    M = kp.shape[0]
    return [bounding_box(kp[i].T, score[i]).reshape(-1) for i in range(M)]

class SkeletonMatcher:
    def __init__(self, iou_threshold, conf_threshold, grace_distance, similarity_threshold, tolerance):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.grace_distance = grace_distance
        self.similarity_threshold = similarity_threshold
        self.tolerance = tolerance

    def _straight_match(self, detections, kp, kps, cids, detected, boxes):
        child_box = None
        for i, df in tqdm(enumerate(detections), desc='Skeleton Matcher'):
            children = df[df['class'] == 1]
            if children.shape[0] == 0:
                continue
            elif children.shape[0] > 1 and child_box is not None:
                candidates = [(i, get_box(b)) for i, b in children.iterrows()]
                ious = [get_iou(get_box(child_box), b) for _, b in candidates]
                child_box = children.loc[candidates[np.argmax(ious)][0]]
                # if not np.equal(child_box.values, children.loc[children['confidence'].idxmax()].values).all():
                #     print('Different child box was chosen!')
            else:
                child_box = children.loc[children['confidence'].idxmax()]
            boxes[i] = get_box(child_box)
            cid, iou = find_nearest(child_box, get_boxes(kp[:, i, :, :], kps[:, i, :]))
            if iou < self.iou_threshold:
                continue
            detected[i] = child_box['confidence']
            cids[i] = cid


    def _interpolate(self, detections, kp, kps, cids, detected, boxes):
        env = [{} for _ in detected]

        def scan(lst, key, reverse):
            tmp = None
            for i, child_conf in reversed(list(enumerate(lst))) if reverse else enumerate(lst):
                if child_conf > self.conf_threshold:
                    tmp = i
                env[i][key] = tmp

        scan(detected, 'prev', reverse=False)
        scan(detected, 'next', reverse=True)

        for i, df in tqdm(enumerate(detections), desc='Interpolate'):
            if detected[i] > self.conf_threshold:
                continue
            prev, next = env[i]['prev'], env[i]['next']
            if not ((prev and np.abs(prev - i) < self.grace_distance) or (next and np.abs(next - i) < self.grace_distance)):
                continue
            j = prev if next is None else next if prev is None else prev if abs(i - prev) >= abs(next - i) else next
            _df = detections[j]
            children = _df[_df['class'] == 1]
            child_box = children.loc[children['confidence'].idxmax()]
            candidate, candidate_iou = find_nearest(child_box, get_boxes(kp[:, i, :, :], kps[:, i, :]))
            if candidate_iou < self.iou_threshold:
                continue
            adults = df[df['class'] == 0]
            adults_matches = [(idx, find_nearest(adult_box, get_boxes(kp[:, i, :, :], kps[:, i, :]))) for idx, adult_box in adults.iterrows()]
            conflicts = [(idx, a, iou) for idx, (a, iou) in adults_matches if a == candidate and iou > self.iou_threshold]
            if any(rival_iou > candidate_iou and \
                   not get_iou(get_box(child_box), get_box(adults.loc[idx])) > self.similarity_threshold for idx, _, rival_iou in conflicts):
                continue
            cids[i] = candidate
            boxes[i] = get_box(child_box)


    def match_skeleton(self, skeleton, detections):
        skeleton = skeleton.copy()
        kp = skeleton['keypoint']
        kps = skeleton['keypoint_score']
        _, T, _, _ = kp.shape
        adj = len(detections) - T
        if np.abs(adj) > self.tolerance:
            raise IndexError(f'Length mismatch: skeleton({T}) - video({len(detections)})')
        # if adj <= 0:
        #     detections = detections + [detections[-1]] * np.abs(adj)
        # else:
        #     detections = detections[adj:]

        skeleton['child_ids'] = np.ones(T) * -1
        skeleton['child_detected'] = np.zeros(T)
        skeleton['child_bbox'] = np.zeros((T, 4))

        cids = skeleton['child_ids']
        detected = skeleton['child_detected']
        boxes = skeleton['child_bbox']
        _, detections = list(zip(*detections))
        self._straight_match(detections, kp, kps, cids, detected, boxes)
        self._interpolate(detections, kp, kps, cids, detected, boxes)
        return skeleton
