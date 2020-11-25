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
Transformer hub interface for transformer large
'''
from src.transformer_model import TransformerModel
from src.transformer_model import TransformerConfig
import mindspore.common.dtype as mstype
transformer_net_cfg_large = TransformerConfig(
    batch_size=96,
    seq_length=128,
    vocab_size=36560,
    hidden_size=1024,
    num_hidden_layers=6,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_act="relu",
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
    max_position_embeddings=128,
    initializer_range=0.02,
    label_smoothing=0.1,
    dtype=mstype.float32,
    compute_type=mstype.float16
)
def create_network(name, *args, **kwargs):
    '''
    Create transformer network for large.
    '''
    if name == 'transformer_large':
        if "batch_size" in kwargs:
            transformer_net_cfg_large.batch_size = kwargs["batch_size"]
        if "seq_length" in kwargs:
            transformer_net_cfg_large.seq_length = kwargs["seq_length"]
        if "vocab_size" in kwargs:
            transformer_net_cfg_large.vocab_size = kwargs["vocab_size"]
        is_training = kwargs.get("is_training", False)
        if not is_training:
            transformer_net_cfg_large.batch_size = 1
            transformer_net_cfg_large.hidden_dropout_prob = 0.
            transformer_net_cfg_large.attention_probs_dropout_prob = 0.
        return TransformerModel(transformer_net_cfg_large, is_training, *args)
    raise NotImplementedError(f"{name} is not implemented in the repo")
