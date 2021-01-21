# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed


config1 = ed({
    # dataset metadata
    "dataset_name": "fsns",
    # annotation files paths
    "train_annotation_file": "path-to-file",
    "test_annotation_file": "path-to-file",
    # dataset root paths
    "data_root_train": "path-to-dir",
    "data_root_test": "path-to-dir",
    # mindrecord target locations
    "mindrecord_dir": "path-to-dir",
    # training and testing params
    "batch_size": 8,
    "epoch_size": 5,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_steps": 2500,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 20,
    "save_checkpoint_path": "./",
    "warmup_epochs": 5,
    "lr_decay_mode": "poly",
    "lr": 1e-4,
    "work_nums": 4,
    "im_size_w": 512,
    "im_size_h": 64,
    "pos_samples_size": 100,
    "augment_severity": 0.1,
    "augment_prob": 0.3
})
