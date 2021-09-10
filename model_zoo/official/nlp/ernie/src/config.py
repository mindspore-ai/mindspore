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
network config setting, will be used in dataset.py, run_pretrain.py
"""
from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .ernie_model import ErnieConfig
cfg = edict({
    'batch_size': 8192,
    'ernie_network': 'base',
    'loss_scale_value': 102400,
    'scale_factor': 2,
    'scale_window': 1000,
    'optimizer': 'AdamWeightDecay',
    'enable_global_norm': False,
    'AdamWeightDecay': edict({
        'learning_rate': 1e-4,
        'end_learning_rate': 0.0,
        'power': 5.0,
        'weight_decay': 1e-2,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
        'eps': 1e-6,
        'warmup_steps': 5000,
    }),
    'Lamb': edict({
        'learning_rate': 3e-4,
        'end_learning_rate': 0.0,
        'power': 2.0,
        'warmup_steps': 10000,
        'weight_decay': 0.01,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
        'eps': 1e-8,
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }),
})

'''
Including two kinds of network: \
base: Baidu ERNIE-base(the base version of ERNIE model).
'''
cfg.batch_size = 64
ernie_net_cfg = ErnieConfig(
    seq_length=512,
    vocab_size=18000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="relu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=513,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    dtype=mstype.float32,
    compute_type=mstype.float16
)
