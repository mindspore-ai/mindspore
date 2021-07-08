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

"""config script"""

from easydict import EasyDict as edict
from mindspore.common import dtype as mstype
from src.model import AlbertConfig


gradient_cfg = edict({
    'clip_type': 1,
    'clip_value': 1.0
})


train_cfg = edict({
    'batch_size': 16,
    'loss_scale_value': 2 ** 16,
    'scale_factor': 2,
    'scale_window': 50,
    'max_global_epoch': 10, #fl_iteration_num
    'server_cfg': edict({
        'learning_rate': 1e-5,
        'max_local_epoch': 1,
        'cyclic_trunc': False
    }),
    'client_cfg': edict({
        'learning_rate': 1e-5,
        'max_local_epoch': 1,
        'num_per_epoch': 20,
        'cyclic_trunc': True
    }),
    'optimizer_cfg': edict({
        'AdamWeightDecay': edict({
            'end_learning_rate': 1e-14,
            'power': 1.0,
            'weight_decay': 1e-4,
            'eps': 1e-6,
            'decay_filter': lambda x: 'norm' not in x.name.lower() and 'bias' not in x.name.lower(),
            'warmup_ratio': 0.1
        }),
    }),
})

eval_cfg = edict({
    'batch_size': 256,
})

server_net_cfg = AlbertConfig(
    seq_length=8,
    vocab_size=11682,
    hidden_size=312,
    num_hidden_groups=1,
    num_hidden_layers=4,
    inner_group_num=1,
    num_attention_heads=12,
    intermediate_size=1248,
    hidden_act="gelu",
    query_act=None,
    key_act=None,
    value_act=None,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    classifier_dropout_prob=0.0,
    embedding_size=128,
    layer_norm_eps=1e-12,
    has_attention_mask=True,
    do_return_2d_tensor=True,
    use_one_hot_embeddings=False,
    use_token_type=True,
    return_all_encoders=False,
    output_attentions=False,
    output_hidden_states=False,
    dtype=mstype.float32,
    compute_type=mstype.float32,
    is_training=True,
    num_labels=4,
    use_word_embeddings=True
)

client_net_cfg = AlbertConfig(
    seq_length=8,
    vocab_size=11682,
    hidden_size=312,
    num_hidden_groups=1,
    num_hidden_layers=4,
    inner_group_num=1,
    num_attention_heads=12,
    intermediate_size=1248,
    hidden_act="gelu",
    query_act=None,
    key_act=None,
    value_act=None,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    classifier_dropout_prob=0.0,
    embedding_size=128,
    layer_norm_eps=1e-12,
    has_attention_mask=True,
    do_return_2d_tensor=True,
    use_one_hot_embeddings=False,
    use_token_type=True,
    return_all_encoders=False,
    output_attentions=False,
    output_hidden_states=False,
    dtype=mstype.float32,
    compute_type=mstype.float32,
    is_training=True,
    num_labels=4,
    use_word_embeddings=True
)
