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
python config.py
"""
from easydict import EasyDict

cfg = EasyDict({
    "class_num": 1000,
    "train_batch": 64,
    "test_batch": 100,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 1,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 5,
    "lr": 0.1,
    "gamma": 0.1,
    "schedule": [30, 60, 90]
})
