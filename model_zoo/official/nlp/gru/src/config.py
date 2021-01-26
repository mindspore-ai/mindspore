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
"""GRU config"""
from easydict import EasyDict

config = EasyDict({
    "batch_size": 16,
    "eval_batch_size": 1,
    "src_vocab_size": 8154,
    "trg_vocab_size": 6113,
    "encoder_embedding_size": 256,
    "decoder_embedding_size": 256,
    "hidden_size": 512,
    "max_length": 32,
    "num_epochs": 30,
    "save_checkpoint": True,
    "ckpt_epoch": 10,
    "target_file": "target.txt",
    "output_file": "output.txt",
    "keep_checkpoint_max": 30,
    "base_lr": 0.001,
    "warmup_step": 300,
    "momentum": 0.9,
    "init_loss_scale_value": 1024,
    'scale_factor': 2,
    'scale_window': 2000,
    "warmup_ratio": 1/3.0,
    "teacher_force_ratio": 0.5
})
