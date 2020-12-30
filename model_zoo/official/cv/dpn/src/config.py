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
from easydict import EasyDict as edict

# config for dpn,imagenet-1K
config = edict()

# model config
config.image_size = (224, 224)  # inpute image size
config.num_classes = 1000  # dataset class number
config.backbone = 'dpn92'  # backbone network
config.is_save_on_master = True

# parallel config
config.num_parallel_workers = 4  # number of workers to read the data
config.rank = 0  # local rank of distributed
config.group_size = 1  # group size of distributed

# training config
config.batch_size = 32  # batch_size
config.global_step = 0  # start step of learning rate
config.epoch_size = 180  # epoch_size
config.loss_scale_num = 1024  # loss scale
# optimizer config
config.momentum = 0.9  # momentum (SGD)
config.weight_decay = 1e-4  # weight_decay (SGD)
# learning rate config
config.lr_schedule = 'warmup'  # learning rate schedule
config.lr_init = 0.01  # init learning rate
config.lr_max = 0.1  # max learning rate
config.factor = 0.1  # factor of lr to drop
config.epoch_number_to_drop = [5, 15]  # learing rate will drop after these epochs
config.warmup_epochs = 5  # warmup epochs in learning rate schedule

# dataset config
config.dataset = "imagenet-1K"  # dataset
config.label_smooth = False  # label_smooth
config.label_smooth_factor = 0.0  # label_smooth_factor

# parameter save config
config.keep_checkpoint_max = 3  # only keep the last keep_checkpoint_max checkpoint
