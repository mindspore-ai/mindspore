# Copyright 2020 Huawei Technologies Co., Ltd
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
    'num_classes': 1,
    'num_joints': 17,
    'max_objs': 32,
    'input_res': [512, 512],
    'output_res': [128, 128],
    'rand_crop': False,
    'shift': 0.1,
    'scale': 0.4,
    'aug_rot': 0.0,
    'rotate': 0,
    'flip_prop': 0.5,
    'mean': np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32),
    'std': np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32),
    'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
    'edges': [[0, 1], [0, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 6],
              [5, 7], [7, 9], [6, 8], [8, 10], [6, 12], [5, 11], [11, 12],
              [12, 14], [14, 16], [11, 13], [13, 15]],
    'eig_val': np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32),
    'eig_vec': np.array([[-0.58752847, -0.69563484, 0.41340352],
                         [-0.5832747, 0.00994535, -0.81221408],
                         [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32),
    'categories': [{"supercategory": "person",
                    "id": 1,
                    "name": "person",
                    "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                                  "left_knee", "right_knee", "left_ankle", "right_ankle"],
                    "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                                 [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                                 [2, 4], [3, 5], [4, 6], [5, 7]]}],
})


net_config = edict({
    'down_ratio': 4,
    'last_level': 6,
    'final_kernel': 1,
    'stage_levels': [1, 1, 1, 2, 2, 1],
    'stage_channels': [16, 32, 64, 128, 256, 512],
    'head_conv': 256,
    'dense_hp': True,
    'hm_hp': True,
    'reg_hp_offset': True,
    'reg_offset': True,
    'hm_weight': 1,
    'off_weight': 1,
    'wh_weight': 0.1,
    'hp_weight': 1,
    'hm_hp_weight': 1,
    'mse_loss': False,
    'reg_loss': 'l1',
})


train_config = edict({
    'batch_size': 32,
    'loss_scale_value': 1024,
    'optimizer': 'Adam',
    'lr_schedule': 'MultiDecay',
    'Adam': edict({
        'weight_decay': 0.0,
        'decay_filter': lambda x: x.name.endswith('.bias') or x.name.endswith('.beta') or x.name.endswith('.gamma'),
    }),
    'PolyDecay': edict({
        'learning_rate': 1.2e-4,
        'end_learning_rate': 5e-7,
        'power': 5.0,
        'eps': 1e-7,
        'warmup_steps': 2000,
    }),
    'MultiDecay': edict({
        'learning_rate': 1.2e-4,
        'eps': 1e-7,
        'warmup_steps': 2000,
        'multi_epochs': [270, 300],
        'factor': 10,
    })
})


eval_config = edict({
    'soft_nms': True,
    'keep_res': True,
    'multi_scales': [1.0],
    'pad': 31,
    'K': 100,
    'score_thresh': 0.3
})


export_config = edict({
    'input_res': dataset_config.input_res,
    'ckpt_file': "./ckpt_file.ckpt",
    'export_format': "MINDIR",
    'export_name': "CenterNet_MultiPose",
})
