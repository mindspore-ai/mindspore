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
network config setting, will be used in train.py and evaluation.py
"""
from easydict import EasyDict as ed

config = ed({
    "learning_rate": 0.0014,
    "weight_decay": 0.00005,
    "momentum": 0.97,
    "crop_size": 513,
    "eval_scales": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    "atrous_rates": None,
    "image_pyramid": None,
    "output_stride": 16,
    "fine_tune_batch_norm": False,
    "ignore_label": 255,
    "decoder_output_stride": None,
    "seg_num_classes": 21,
    "epoch_size": 6,
    "batch_size": 2,
    "enable_save_ckpt": True,
    "save_checkpoint_steps": 10000,
    "save_checkpoint_num": 1
})
