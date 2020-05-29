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
config settings, will be used in finetune.py
"""

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .bert_model import BertConfig

cfg = edict({
    'task': 'NER',
    'num_labels': 41,
    'data_file': '/your/path/evaluation.tfrecord',
    'schema_file': '/your/path/schema.json',
    'finetune_ckpt': '/your/path/your.ckpt',
    'use_crf': False,
    'clue_benchmark': False,
})

bert_net_cfg = BertConfig(
    batch_size=16 if not cfg.clue_benchmark else 1,
    seq_length=128,
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    input_mask_from_dataset=True,
    token_type_ids_from_dataset=True,
    dtype=mstype.float32,
    compute_type=mstype.float16,
)
