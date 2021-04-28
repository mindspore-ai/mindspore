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
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

config = ed({
    "save_checkpoint": True,
    "save_checkpoint_epochs": 112,
    "keep_checkpoint_max": 10000,
    "learning_rate": 0.001,
    "m_for_scrutinizer": 4,
    "topK": 6,
    "input_size": (448, 448),
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "num_epochs": 112,
    "num_classes": 200,
    "num_train_images": 5994,
    "num_test_images": 5794,
    "batch_size": 8,
    "prefix": "ntsnet",
    "lossLogName": "loss.log"
})
