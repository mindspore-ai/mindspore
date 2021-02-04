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
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed
# config optimizer for resnet50, imagenet2012. Momentum is default, Thor is optional.
cfg = ed({
    'optimizer': 'Momentum',
    })

# config for resent50, cifar10
config1 = ed({
    "class_num": 10,
    "batch_size": 32,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 90,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 5,
    "lr_decay_mode": "poly",
    "lr_init": 0.01,
    "lr_end": 0.00001,
    "lr_max": 0.1
})

# config for resnet50, imagenet2012
config2 = ed({
    "class_num": 1001,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 90,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 0,
    "lr_decay_mode": "linear",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0,
    "lr_max": 0.8,
    "lr_end": 0.0
})

# config for resent101, imagenet2012
config3 = ed({
    "class_num": 1001,
    "batch_size": 32,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 120,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 0,
    "lr_decay_mode": "cosine",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr": 0.1
})

# config for se-resnet50, imagenet2012
config4 = ed({
    "class_num": 1001,
    "batch_size": 32,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 28,
    "train_epoch_size": 24,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 4,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 3,
    "lr_decay_mode": "cosine",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.0,
    "lr_max": 0.3,
    "lr_end": 0.0001
})

# config for resnet50, imagenet2012, Ascend 910
config_thor_Ascend = ed({
    "class_num": 1001,
    "batch_size": 32,
    "loss_scale": 128,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epoch_size": 45,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 2,
    "keep_checkpoint_max": 15,
    "save_checkpoint_path": "./",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.05803,
    "lr_decay": 4.04839,
    "lr_end_epoch": 53,
    "damping_init": 0.02714,
    "damping_decay": 0.50036,
    "frequency": 834,
})

# config for resnet50, imagenet2012, GPU
config_thor_gpu = ed({
    "class_num": 1001,
    "batch_size": 32,
    "loss_scale": 128,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epoch_size": 40,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 15,
    "save_checkpoint_path": "./",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.05672,
    "lr_decay": 4.9687,
    "lr_end_epoch": 50,
    "damping_init": 0.02345,
    "damping_decay": 0.5467,
    "frequency": 834,
})
