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
network config setting
"""
from easydict import EasyDict as edict

common_config = edict({
    'device_id': 0,
    'pre_trained': True,
    'max_steps': 30000,
    'save_checkpoint': True,
    # 'pre_trained_file': '/home/work/user-job-dir/posenet/pre_trained_googlenet_imagenet.ckpt',
    'pre_trained_file': '../pre_trained_googlenet_imagenet.ckpt',
    'checkpoint_dir': '../checkpoint',
    'save_checkpoint_epochs': 5,
    'keep_checkpoint_max': 10
})

KingsCollege = edict({
    'batch_size': 75,
    'lr_init': 0.001,
    'weight_decay': 0.5,
    'name': 'KingsCollege',
    'dataset_path': '../KingsCollege/',
    'mindrecord_dir': '../MindrecordKingsCollege'
})

StMarysChurch = edict({
    'batch_size': 75,
    'lr_init': 0.001,
    'weight_decay': 0.5,
    'name': 'StMarysChurch',
    'dataset_path': '../StMarysChurch/',
    'mindrecord_dir': '../MindrecordStMarysChurch'
})
