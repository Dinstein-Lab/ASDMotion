class GraphLayout:
    def __init__(self, name, center, joints, pairs, face=False, hand=False, model_pose=None):
        self.name = name
        self.center = center
        self._pose_joints = joints
        self._pose_pairs = pairs
        self._len = len(joints)
        self._pose_map = {}
        self.model_pose = model_pose if model_pose else name
        self.face = face
        self.hand = hand
        for k, v in joints.items():
            self._pose_map[k] = v
            self._pose_map[v] = k

    def face_joints(self):
        return [k for k in self._pose_map.keys() if type(k) == str and any(s in k for s in ['Eye', 'Ear', 'Nose'])]

    def __len__(self):
        return self._len

    def joint(self, i):
        return self._pose_map[i]

    def joints(self):
        return self._pose_joints

    def pairs(self):
        return self._pose_pairs

    def neighbors(self, i):
        return []


def convert_layout(np_data, l1, l2):
    assert len(l2) <= len(l1)
    joints = [l1.joint(i) for i in l2.joints().values()]
    return np_data[joints]


BODY_25_LAYOUT = GraphLayout(
    'BODY_25',
    1,
    {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "MidHip",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "LBigToe",
        20: "LSmallToe",
        21: "LHeel",
        22: "RBigToe",
        23: "RSmallToe",
        24: "RHeel"
    }, [(0, 1), (1, 8),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11), (11, 22), (11, 24), (22, 23),
        (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20),
        (0, 15), (15, 17), (0, 16), (16, 18)]
)

BODY_21_LAYOUT = GraphLayout(
    'BODY_21',
    1,
    {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "LowerAbs",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "RealNeck",
        20: "Top"
    }, [(1, 8), (0, 1), (1, 19), (19, 20),
        (0, 15), (15, 17),
        (0, 16), (16, 18),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (2, 9), (8, 9), (9, 10), (10, 11),
        (5, 12), (8, 12), (12, 13), (13, 14),
        (2, 17), (5, 18)
        ]
)

COCO_LAYOUT = GraphLayout(
    'COCO',
    1,
    {
        0: "Nose",
        1: "LEye",
        2: "REye",
        3: "LEar",
        4: "REar",
        5: "LShoulder",
        6: "RShoulder",
        7: "LElbow",
        8: "RElbow",
        9: "LWrist",
        10: "RWrist",
        11: "LHip",
        12: "RHip",
        13: "LKnee",
        14: "RKnee",
        15: "LAnkle",
        16: "RAnkle"
    }, [(15, 13), (13, 11), (16, 14),
        (14, 12), (11, 12), (5, 11), (6, 12),
        (5, 6), (7, 5), (8, 6), (9, 7),
        (10, 8), (1, 2), (1, 0), (2, 0),
        (3, 1), (4, 2), (3, 5), (4, 6)]
)
