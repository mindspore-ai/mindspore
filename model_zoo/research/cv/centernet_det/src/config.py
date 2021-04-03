# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
network config setting, will be used in dataset.py, train.py， eval.py
"""

import numpy as np
from easydict import EasyDict as edict


dataset_config = edict({
    "num_classes": 80,
    'max_objs': 128,
    'input_res': [512, 512],
    'output_res': [128, 128],
    'rand_crop': True,
    'shift': 0.1,
    'scale': 0.4,
    'down_ratio': 4,
    'aug_rot': 0.0,
    'rotate': 0,
    'flip_prop': 0.5,
    'color_aug': True,
    'coco_classes': ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush'),
    'coco_class_name2id': {
        'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
        'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11,
        'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19,
        'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
        'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35,
        'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40,
        'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'wine glass': 46,
        'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54,
        'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61,
        'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72,
        'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78,
        'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86,
        'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90},
    'mean': np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32),
    'std': np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32),
    'eig_val': np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32),
    'eig_vec': np.array([[-0.58752847, -0.69563484, 0.41340352],
                         [-0.5832747, 0.00994535, -0.81221408],
                         [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32),
})


net_config = edict({
    'down_ratio': 4,
    'last_level': 6,
    'num_stacks': 2,
    'n': 5,
    'heads': {'hm': 80, 'wh': 2, 'reg': 2},
    'cnv_dim': 256,
    'modules': [2, 2, 2, 2, 2, 4],
    'dims': [256, 256, 384, 384, 384, 512],
    'dense_wh': False,
    'norm_wh': False,
    'cat_spec_wh': False,
    'reg_offset': True,
    'hm_weight': 1,
    'off_weight': 1,
    'wh_weight': 0.1,
    'mse_loss': False,
    'reg_loss': 'l1',
})


train_config = edict({
    'batch_size': 12,
    'loss_scale_value': 1024,
    'optimizer': 'Adam',
    'lr_schedule': 'MultiDecay',
    'Adam': edict({
        'weight_decay': 0.0,
        'decay_filter': lambda x: x.name.endswith('.bias') or x.name.endswith('.beta') or x.name.endswith('.gamma'),
    }),
    'PolyDecay': edict({
        'learning_rate': 2.4e-4,
        'end_learning_rate': 2.4e-7,
        'power': 5.0,
        'eps': 1e-7,
        'warmup_steps': 2000,
    }),
    'MultiDecay': edict({
        'learning_rate': 2.4e-4,
        'eps': 1e-7,
        'warmup_steps': 2000,
        'multi_epochs': [105, 125],
        'factor': 10,
    })
})


eval_config = edict({
    'SOFT_NMS': True,
    'keep_res': True,
    'multi_scales': [1.0],
    'pad': 127,
    'K': 100,
    'score_thresh': 0.3,
    'valid_ids': [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90],
    'color_list': [
        0.000, 0.800, 1.000,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.333,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.800, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.667, 0.400,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.200, 0.800,
        0.143, 0.143, 0.543,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.50, 0.5, 0],
})

export_config = edict({
    'input_res': dataset_config.input_res,
    'ckpt_file': "./ckpt_file.ckpt",
    'export_format': "MINDIR",
    'export_name': "CenterNet_ObjectDetection",
})
