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
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict


FCN8s_VOC2012_cfg = edict({
    # dataset
    'data_file': '/data/workspace/mindspore_dataset/FCN/FCN/dataset/MINDRECORED_NAME.mindrecord',
    'batch_size': 32,
    'crop_size': 512,
    'image_mean': [103.53, 116.28, 123.675],
    'image_std': [57.375, 57.120, 58.395],
    'min_scale': 0.5,
    'max_scale': 2.0,
    'ignore_label': 255,
    'num_classes': 21,

    # optimizer
    'train_epochs': 500,
    'base_lr': 0.015,
    'loss_scale': 1024.0,

    # model
    'model': 'FCN8s',
    'ckpt_vgg16': '/data/workspace/mindspore_dataset/FCN/FCN/model/0-150_5004.ckpt',
    'ckpt_pre_trained': '/data/workspace/mindspore_dataset/FCN/FCN/model_new/FCN8s-500_82.ckpt',

    # train
    'save_steps': 330,
    'keep_checkpoint_max': 500,
    'train_dir': '/data/workspace/mindspore_dataset/FCN/FCN/model_new/',
})
