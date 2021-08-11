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
lightcnn_cfg = edict({
    # training setting
    'network_type': 'LightCNN_9Layers',
    'epochs': 80,
    'lr': 0.01,
    'num_classes': 79077,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'batch_size': 128,
    'image_size': 128,
    'save_checkpoint_steps': 60000,
    'keep_checkpoint_max': 40,
    # train data location
    'data_path': '/opt/data/lightcnn_data/FaceImageCroppedWithAlignment/',
    'train_list': '/opt/data/lightcnn_data/MS-Celeb-1M_clean_list.txt',
    # test data location
    'root_path': '/opt/data/lightcnn_data/lfw/image',
    'lfw_img_list': 'image_list_for_lfw.txt',
    'lfw_pairs_mat_path': 'mat_files/lfw_pairs.mat',
    'blufr_img_list': 'image_list_for_blufr.txt',
    'blufr_config_mat_path': 'mat_files/blufr_lfw_config.mat'
})
