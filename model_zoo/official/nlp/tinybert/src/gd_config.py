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
network config setting, will be used in dataset.py, run_general_distill.py and run_task_distill.py
"""
from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .tinybert_model import BertConfig

common_cfg = edict({
    'batch_size': 32,
    'loss_scale_value': 2 ** 16,
    'scale_factor': 2,
    'scale_window': 1000,
    'AdamWeightDecay': edict({
        'learning_rate': 5e-5,
        'end_learning_rate': 1e-14,
        'power': 1.0,
        'weight_decay': 1e-4,
        'eps': 1e-6,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
    }),
})
'''
Including two kinds of network: \
teacher network: The BERT-base network.
student network: The network which is inherited from teacher network.
'''
bert_teacher_net_cfg = BertConfig(
    seq_length=128,
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    dtype=mstype.float32,
    compute_type=mstype.float16
)
bert_student_net_cfg = BertConfig(
    seq_length=128,
    vocab_size=30522,
    hidden_size=384,
    num_hidden_layers=4,
    num_attention_heads=12,
    intermediate_size=1536,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    dtype=mstype.float32,
    compute_type=mstype.float16
)
