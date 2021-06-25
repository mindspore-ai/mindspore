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
"""Configure"""
from easydict import EasyDict as ed

config_zero_shot_standalone = ed({
    "dp": 1,
    "mp": 1,
    "batch_size": 1,
    "rank_size": 1,
    "vocab_size": 30000,
    'seq_length': 571,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
})

config_zero_shot_distrubute = ed({
    "dp": 1,
    "mp": 2,
    "batch_size": 2,
    "rank_size": 2,
    "vocab_size": 30000,
    'seq_length': 571,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
})

finetune_dev_standalone = ed({
    "dp": 1,
    "mp": 1,
    "batch_size": 1,
    "rank_size": 1,
    "vocab_size": 30000,
    'seq_length': 696,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
})

finetune_dev_distrubute = ed({
    "dp": 1,
    "mp": 2,
    "batch_size": 1,
    "rank_size": 2,
    "vocab_size": 30000,
    'seq_length': 696,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
})

finetune_test_standalone = ed({
    "dp": 1,
    "mp": 1,
    "batch_size": 1,
    "rank_size": 1,
    "vocab_size": 30000,
    'seq_length': 666,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
})

finetune_test_distrubute = ed({
    "dp": 1,
    "mp": 2,
    "batch_size": 1,
    "rank_size": 2,
    "vocab_size": 30000,
    'seq_length': 666,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
})

config_train_single_machine = ed({
    "dp": 4,
    "mp": 2,
    "epoch": 10,
    "batch_size": 16,
    "rank_size": 8,
    "vocab_size": 30000,
    'seq_length': 725,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "lr": 1e-5,
    "eps": 1e-8,
    "dropout": 0.2,
    "end_learning_rate": 1e-7,
    "weight_decay": 1e-2,
    "warmup_steps": 0.05,
    "power": 1.0,
    "grad_accumulation_step": 4,
    "sink_size": 1
})

config_train_multi_machine = ed({
    "dp": 16,
    "mp": 2,
    "epoch": 10,
    "batch_size": 128,
    "rank_size": 32,
    "vocab_size": 30000,
    'seq_length': 725,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "lr": 2e-5,
    "eps": 1e-8,
    "dropout": 0.1,
    "end_learning_rate": 1e-7,
    "weight_decay": 1e-2,
    "warmup_steps": 0.1,
    "power": 1.0,
    "grad_accumulation_step": 1,
    "sink_size": 1
})
