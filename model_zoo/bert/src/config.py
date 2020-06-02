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
network config setting, will be used in dataset.py, run_pretrain.py
"""
from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .bert_model import BertConfig
cfg = edict({
    'bert_network': 'base',
    'loss_scale_value': 65536,
    'scale_factor': 2,
    'scale_window': 1000,
    'optimizer': 'Lamb',
    'AdamWeightDecayDynamicLR': edict({
        'learning_rate': 3e-5,
        'end_learning_rate': 1e-10,
        'power': 5.0,
        'weight_decay': 1e-5,
        'eps': 1e-6,
        'warmup_steps': 10000,
    }),
    'Lamb': edict({
        'start_learning_rate': 3e-5,
        'end_learning_rate': 1e-10,
        'power': 10.0,
        'warmup_steps': 10000,
        'weight_decay': 0.01,
        'eps': 1e-6,
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }),
})

'''
Including two kinds of network: \
base: Goole BERT-base(the base version of BERT model).
large: BERT-NEZHA(a Chinese pretrained language model developed by Huawei, which introduced a improvement of \
       Functional Relative Posetional Encoding as an effective positional encoding scheme).
'''
if cfg.bert_network == 'base':
    bert_net_cfg = BertConfig(
        batch_size=32,
        seq_length=128,
        vocab_size=21128,
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
        input_mask_from_dataset=True,
        token_type_ids_from_dataset=True,
        dtype=mstype.float32,
        compute_type=mstype.float16
    )
if cfg.bert_network == 'nezha':
    bert_net_cfg = BertConfig(
        batch_size=32,
        seq_length=128,
        vocab_size=21128,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        use_relative_positions=True,
        input_mask_from_dataset=True,
        token_type_ids_from_dataset=True,
        dtype=mstype.float32,
        compute_type=mstype.float16
    )
if cfg.bert_network == 'large':
    bert_net_cfg = BertConfig(
        batch_size=16,
        seq_length=512,
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        use_relative_positions=False,
        input_mask_from_dataset=True,
        token_type_ids_from_dataset=True,
        dtype=mstype.float32,
        compute_type=mstype.float16,
        enable_fused_layernorm=True
    )
