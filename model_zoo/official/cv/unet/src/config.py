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

cfg_unet_medical = {
    'model': 'unet_medical',
    'crop': [388 / 572, 388 / 572],
    'img_size': [572, 572],
    'lr': 0.0001,
    'epochs': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
}

cfg_unet_nested = {
    'model': 'unet_nested',
    'crop': None,
    'img_size': [576, 576],
    'lr': 0.0001,
    'epochs': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
}

cfg_unet_simple = {
    'model': 'unet_simple',
    'crop': None,
    'img_size': [576, 576],
    'lr': 0.0001,
    'epochs': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
}

cfg_unet = cfg_unet_medical
