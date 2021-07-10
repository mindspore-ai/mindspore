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
network config setting
"""
from easydict import EasyDict

resize_value = 224    # image resize

basic_config = EasyDict({
    'random_seed': 1
})

efficientnet_b0_config_cifar10 = EasyDict({
    'model': 'efficientnet_b0',
    'drop': 0.2,
    'drop_connect': 0.2,
    'opt_eps': 0.0001,
    'lr': 0.0002,
    'batch_size': 32,
    'decay_epochs': 2.4,
    'warmup_epochs': 5,
    'decay_rate': 0.97,
    'weight_decay': 1e-5,
    'epochs': 150,
    'workers': 8,
    'amp_level': 'O0',
    'opt': 'rmsprop',
    'num_classes': 10,
    #'Type of global pool, "avg", "max", "avgmax", "avgmaxc"
    'gp': 'avg',
    'momentum': 0.9,
    'warmup_lr_init': 0.0001,
    'smoothing': 0.1,
    #Use Tensorflow BatchNorm defaults for models that support it
    'bn_tf': False,
    'save_checkpoint': True,
    'keep_checkpoint_max': 10,
    'loss_scale': 1024,
    'resume_start_epoch': 0,
})


efficientnet_b0_config_imagenet = EasyDict({
    'model': 'efficientnet_b0',
    'drop': 0.2,
    'drop_connect': 0.2,
    'opt_eps': 0.001,
    'lr': 0.064,
    'batch_size': 128,
    'decay_epochs': 2.4,
    'warmup_epochs': 5,
    'decay_rate': 0.97,
    'weight_decay': 1e-5,
    'epochs': 600,
    'workers': 8,
    'amp_level': 'O0',
    'opt': 'rmsprop',
    'num_classes': 1000,
    #'Type of global pool, "avg", "max", "avgmax", "avgmaxc"
    'gp': 'avg',
    'momentum': 0.9,
    'warmup_lr_init': 0.0001,
    'smoothing': 0.1,
    #Use Tensorflow BatchNorm defaults for models that support it
    'bn_tf': False,
    'save_checkpoint': True,
    'keep_checkpoint_max': 10,
    'loss_scale': 1024,
    'resume_start_epoch': 0,
})

dataset_config = {
    "imagenet": EasyDict({
        "size": (224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "cfg": efficientnet_b0_config_imagenet
    }),

    "cifar10": EasyDict({
        "size": (32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.247, 0.2435, 0.2616),
        "cfg": efficientnet_b0_config_cifar10
    })
}
