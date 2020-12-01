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
# ===========================================================================
"""Network config setting, will be used in train.py and eval.py"""
from easydict import EasyDict as ed

config = ed({
    'batch_size': 64,
    'warmup_lr': 0.0004,
    'lr_rates': [0.002, 0.004, 0.002, 0.0008, 0.0004, 0.0002, 0.00008, 0.00004, 0.000004],
    'lr_steps': [1000, 10000, 40000, 60000, 80000, 100000, 130000, 160000, 190000],
    'gamma': 0.5,
    'weight_decay': 0.0005,
    'momentum': 0.5,
    'max_epoch': 2500,

    'log_interval': 10,
    'ckpt_path': '../../output',
    'ckpt_interval': 1000,
    'result_path': '../../results',

    'input_shape': [768, 448],
    'jitter': 0.3,
    'flip': 0.5,
    'hue': 0.1,
    'sat': 1.5,
    'val': 1.5,
    'num_classes': 1,
    'anchors': [
        [3, 4],
        [5, 6],
        [7, 9],
        [10, 13],
        [15, 19],
        [21, 26],
        [28, 36],
        [38, 49],
        [54, 71],
        [77, 102],
        [122, 162],
        [207, 268],
    ],
    'anchors_mask': [(8, 9, 10, 11), (4, 5, 6, 7), (0, 1, 2, 3)],

    'conf_thresh': 0.1,
    'nms_thresh': 0.45,
})
