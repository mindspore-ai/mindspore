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
'''
Bert hub interface for bert base and bert nezha
'''
from src.bert_model import BertModel
from src.bert_model import BertConfig
import mindspore.common.dtype as mstype
bert_net_cfg_base = BertConfig(
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
    dtype=mstype.float32,
    compute_type=mstype.float16
)
bert_net_cfg_nezha = BertConfig(
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
    dtype=mstype.float32,
    compute_type=mstype.float16
)
def create_network(name, *args, **kwargs):
    '''
    Create bert network for base and nezha.
    '''
    if name == 'bert_base':
        if "seq_length" in kwargs:
            bert_net_cfg_base.seq_length = kwargs["seq_length"]
        is_training = kwargs.get("is_training", default=False)
        return BertModel(bert_net_cfg_base, is_training, *args)
    if name == 'bert_nezha':
        if "seq_length" in kwargs:
            bert_net_cfg_nezha.seq_length = kwargs["seq_length"]
        is_training = kwargs.get("is_training", default=False)
        return BertModel(bert_net_cfg_nezha, is_training, *args)
    raise NotImplementedError(f"{name} is not implemented in the repo")
