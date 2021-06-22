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
"""Config for train and eval."""
import os

faceboxes_config = {
    # ---------------- train ----------------
    'image_size': (1024, 1024),
    'batch_size': 8,
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'class_weight': 1.0,
    'match_thresh': 0.35,
    'num_worker': 8,

    # checkpoint
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 50,
    "save_checkpoint_path": "./",

    # env
    "device_id": int(os.getenv('DEVICE_ID', '0')),
    "rank_id": int(os.getenv('RANK_ID', '0')),
    "rank_size": int(os.getenv('RANK_SIZE', '1')),

    # seed
    'seed': 1,

    # opt
    'optim': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,

    # lr
    'epoch': 300,
    'decay1': 200,
    'decay2': 250,
    'lr_type': 'dynamic_lr',
    'initial_lr': 0.001,
    'warmup_epoch': 4,
    'gamma': 0.1,

    # ---------------- val ----------------
    'val_model': '../train/rank0/ckpt_0/FaceBoxes-300_402.ckpt',
    'val_dataset_folder': '../data/widerface/val/',
    'val_origin_size': True,
    'val_confidence_threshold': 0.05,
    'val_nms_threshold': 0.4,
    'val_iou_threshold': 0.5,
    'val_save_result': False,
    'val_predict_save_folder': './widerface_result',
    'val_gt_dir': '../data/widerface/val/ground_truth',
}
