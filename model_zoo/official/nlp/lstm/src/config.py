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
from easydict import EasyDict as edict

# LSTM CONFIG
lstm_cfg = edict({
    'num_classes': 2,
    'dynamic_lr': False,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'num_epochs': 20,
    'batch_size': 64,
    'embed_size': 300,
    'num_hiddens': 100,
    'num_layers': 2,
    'bidirectional': True,
    'save_checkpoint_steps': 390,
    'keep_checkpoint_max': 10
})

# LSTM CONFIG IN ASCEND for 1p training
lstm_cfg_ascend = edict({
    'num_classes': 2,
    'momentum': 0.9,
    'num_epochs': 20,
    'batch_size': 64,
    'embed_size': 300,
    'num_hiddens': 128,
    'num_layers': 2,
    'bidirectional': True,
    'save_checkpoint_steps': 7800,
    'keep_checkpoint_max': 10,
    'dynamic_lr': True,
    'lr_init': 0.05,
    'lr_end': 0.01,
    'lr_max': 0.1,
    'lr_adjust_epoch': 6,
    'warmup_epochs': 1,
    'global_step': 0
})

# LSTM CONFIG IN ASCEND for 8p training
lstm_cfg_ascend_8p = edict({
    'num_classes': 2,
    'momentum': 0.9,
    'num_epochs': 20,
    'batch_size': 64,
    'embed_size': 300,
    'num_hiddens': 128,
    'num_layers': 2,
    'bidirectional': True,
    'save_checkpoint_steps': 7800,
    'keep_checkpoint_max': 10,
    'dynamic_lr': True,
    'lr_init': 0.05,
    'lr_end': 0.01,
    'lr_max': 0.3,
    'lr_adjust_epoch': 20,
    'warmup_epochs': 2,
    'global_step': 0
})
