import numpy as np

from jordi.utils import EPSILON


def get_box(row):
    return np.array([row['xcenter'], row['ycenter'], row['width'], row['height']])


def find_nearest(child_row, boxes):
    cb = get_box(child_row)
    iou = [get_iou(cb, b) for b in boxes]
    nearest = np.argmax(iou)
    return nearest, np.max(iou)


def get_iou(_bb1, _bb2):
    def convert(bb):
        return {'x1': bb[0] - bb[2] // 2,
                'y1': bb[1] - bb[3] // 2,
                'x2': bb[0] + bb[2] // 2,
                'y2': bb[1] + bb[3] // 2}

    bb1 = convert(_bb1)
    bb2 = convert(_bb2)

    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def box_distance(b1, b2):
    c1, _ = b1
    c2, _ = b2
    return np.linalg.norm(c1 - c2)


def bounding_box(pose, score, epsilon=EPSILON):
    pose, score = np.array(pose), np.array(score)
    if pose.shape[1] == 2:
        pose = pose.T
    x, y = pose[0][score > epsilon], pose[1][score > epsilon]
    if not any(x):
        x = np.array([0])
    if not any(y):
        y = np.array([0])
    w, h = (np.max(x) - np.min(x)), (np.max(y) - np.min(y))
    return np.array([np.min(x) + w / 2, np.min(y) + h / 2, w, h]).reshape((2, 2))
