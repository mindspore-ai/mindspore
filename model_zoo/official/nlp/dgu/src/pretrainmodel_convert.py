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
convert pretrain model from pdparams to mindspore ckpt
"""
import collections
import os
import paddle.fluid.dygraph as D
from paddle import fluid
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's BERT to mindspore's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'bert.embeddings.word_embeddings.weight': "bert.bert.bert_embedding_lookup.embedding_table",
        'bert.embeddings.token_type_embeddings.weight': "bert.bert.bert_embedding_postprocessor.embedding_table",
        'bert.embeddings.position_embeddings.weight': "bert.bert.bert_embedding_postprocessor.full_position_embeddings",
        'bert.embeddings.layer_norm.weight': 'bert.bert.bert_embedding_postprocessor.layernorm.gamma',
        'bert.embeddings.layer_norm.bias': 'bert.bert.bert_embedding_postprocessor.layernorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'bert.encoder.layers.{i}.self_attn.q_proj.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.attention.query_layer.weight'
        weight_map[f'bert.encoder.layers.{i}.self_attn.q_proj.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.attention.query_layer.bias'
        weight_map[f'bert.encoder.layers.{i}.self_attn.k_proj.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.attention.key_layer.weight'
        weight_map[f'bert.encoder.layers.{i}.self_attn.k_proj.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.attention.key_layer.bias'
        weight_map[f'bert.encoder.layers.{i}.self_attn.v_proj.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.attention.value_layer.weight'
        weight_map[f'bert.encoder.layers.{i}.self_attn.v_proj.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.attention.value_layer.bias'
        weight_map[f'bert.encoder.layers.{i}.self_attn.out_proj.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.output.dense.weight'
        weight_map[f'bert.encoder.layers.{i}.self_attn.out_proj.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.output.dense.bias'
        weight_map[f'bert.encoder.layers.{i}.linear1.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.intermediate.weight'
        weight_map[f'bert.encoder.layers.{i}.linear1.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.intermediate.bias'
        weight_map[f'bert.encoder.layers.{i}.linear2.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.output.dense.weight'
        weight_map[f'bert.encoder.layers.{i}.linear2.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.output.dense.bias'
        weight_map[f'bert.encoder.layers.{i}.norm1.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.output.layernorm.gamma'
        weight_map[f'bert.encoder.layers.{i}.norm1.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.attention.output.layernorm.beta'
        weight_map[f'bert.encoder.layers.{i}.norm2.weight'] = \
        f'bert.bert.bert_encoder.layers.{i}.output.layernorm.gamma'
        weight_map[f'bert.encoder.layers.{i}.norm2.bias'] = \
        f'bert.bert.bert_encoder.layers.{i}.output.layernorm.beta'
    # add pooler
    weight_map.update(
        {
            'bert.pooler.dense.weight': 'bert.bert.dense.weight',
            'bert.pooler.dense.bias': 'bert.bert.dense.bias'
        }
    )
    return weight_map

input_dir = '.'
state_dict = []
bert_weight_map = build_params_map(attention_num=12)
with fluid.dygraph.guard():
    paddle_paddle_params, _ = D.load_dygraph(os.path.join(input_dir, 'bert-base-uncased'))
for weight_name, weight_value in paddle_paddle_params.items():
    if 'weight' in weight_name:
        if 'encoder' in weight_name or 'pooler' in weight_name or \
        'predictions' in weight_name or 'seq_relationship' in weight_name:
            weight_value = weight_value.transpose()
    if weight_name in bert_weight_map.keys():
        state_dict.append({'name': bert_weight_map[weight_name], 'data': Tensor(weight_value)})
        print(weight_name, '->', bert_weight_map[weight_name], weight_value.shape)
save_checkpoint(state_dict, 'base-BertCLS-111.ckpt')
