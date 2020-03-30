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
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from mindspore.model_zoo.Bert_NEZHA import BertConfig
bert_train_cfg = edict({
    'epoch_size': 10,
    'num_warmup_steps': 0,
    'start_learning_rate': 1e-4,
    'end_learning_rate': 0.0,
    'decay_steps': 1000,
    'power': 10.0,
    'save_checkpoint_steps': 2000,
    'keep_checkpoint_max': 10,
    'checkpoint_prefix': "checkpoint_bert",
    # please add your own dataset path
    'DATA_DIR': "/your/path/examples.tfrecord",
    # please add your own dataset schema path
    'SCHEMA_DIR': "/your/path/datasetSchema.json"
})
bert_net_cfg = BertConfig(
    batch_size=16,
    seq_length=128,
    vocab_size=21136,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=True,
    input_mask_from_dataset=True,
    token_type_ids_from_dataset=True,
    dtype=mstype.float32,
    compute_type=mstype.float16,
)
