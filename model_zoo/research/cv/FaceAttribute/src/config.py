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
    'per_batch_size': 128,
    'dst_h': 112,
    'dst_w': 112,
    'workers': 8,
    'attri_num': 3,
    'classes': '9,2,2',
    'backbone': 'resnet18',
    'loss_scale': 1024,
    'flat_dim': 512,
    'fc_dim': 256,
    'lr': 0.009,
    'lr_scale': 1,
    'lr_epochs': [20, 30, 50],
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'max_epoch': 70,
    'warmup_epochs': 0,
    'log_interval': 10,
    'ckpt_path': '../../output',

    # data_to_mindrecord parameter
    'eval_dataset_txt_file': 'Your_label_txt_file',
    'eval_mindrecord_file_name': 'Your_output_path/data_test.mindrecord',
    'train_dataset_txt_file': 'Your_label_txt_file',
    'train_mindrecord_file_name': 'Your_output_path/data_train.mindrecord',
    'train_append_dataset_txt_file': 'Your_label_txt_file',
    'train_append_mindrecord_file_name': 'Your_previous_output_path/data_train.mindrecord0'
})
