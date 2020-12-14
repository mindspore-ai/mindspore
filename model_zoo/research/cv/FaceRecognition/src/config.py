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
#" :===========================================================================

"""network config setting, will be used in train.py and eval.py."""

from easydict import EasyDict as edict

config_base = edict({
    # dataset related
    'data_dir': "your_dataset_path",
    'num_classes': 1,
    'per_batch_size': 192,

    # network structure related
    'backbone': 'r100',
    'use_se': 1,
    'emb_size': 512,
    'act_type': 'relu',
    'fp16': 1,
    'pre_bn': 1,
    'inference': 0,
    'use_drop': 1,
    'nc_16': 1,

    # loss related
    'margin_a': 1.0,
    'margin_b': 0.2,
    'margin_m': 0.3,
    'margin_s': 64,

    # optimizer related
    'lr': 0.4,
    'lr_scale': 1,
    'lr_epochs': '8,14,18',
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'max_epoch': 20,
    'pretrained': '',
    'warmup_epochs': 2,

    # distributed parameter
    'is_distributed': 1,
    'local_rank': 0,
    'world_size': 1,
    'model_parallel': 0,

    # logging related
    'log_interval': 100,
    'ckpt_path': 'outputs',
    'max_ckpts': -1,
    'dynamic_init_loss_scale': 65536,
    'ckpt_steps': 1000
})

config_beta = edict({
    # dataset related
    'data_dir': "your_dataset_path",
    'num_classes': 1,
    'per_batch_size': 192,

    # network structure related
    'backbone': 'r100',
    'use_se': 0,
    'emb_size': 256,
    'act_type': 'relu',
    'fp16': 1,
    'pre_bn': 0,
    'inference': 0,
    'use_drop': 1,
    'nc_16': 1,

    # loss related
    'margin_a': 1.0,
    'margin_b': 0.2,
    'margin_m': 0.3,
    'margin_s': 64,

    # optimizer related
    'lr': 0.04,
    'lr_scale': 1,
    'lr_epochs': '8,14,18',
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'max_epoch': 20,
    'pretrained': 'your_pretrained_model',
    'warmup_epochs': 2,

    # distributed parameter
    'is_distributed': 1,
    'local_rank': 0,
    'world_size': 1,
    'model_parallel': 0,

    # logging related
    'log_interval': 100,
    'ckpt_path': 'outputs',
    'max_ckpts': -1,
    'dynamic_init_loss_scale': 65536,
    'ckpt_steps': 1000
})


config_inference = edict({
    # distributed parameter
    'is_distributed': 0,
    'local_rank': 0,
    'world_size': 1,

    # test weight
    'weight': 'your_test_model',
    'test_dir': 'your_dataset_path',

    # model define
    'backbone': 'r100',
    'use_se': 0,
    'emb_size': 256,
    'act_type': 'relu',
    'fp16': 1,
    'pre_bn': 0,
    'inference': 1,
    'use_drop': 0,

    # test and dis batch size
    'test_batch_size': 128,
    'dis_batch_size': 512,

    # log
    'log_interval': 100,
    'ckpt_path': 'outputs/models',

    # test and dis image list
    'test_img_predix': '',
    'test_img_list': '',
    'dis_img_predix': '',
    'dis_img_list': ''
})
