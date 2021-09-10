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
convert paddle weights to mindspore
"""
import collections
import os
import json
import shutil
import argparse
import paddle.fluid.dygraph as D
from paddle import fluid
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's ernie
    :return:
    """
    weight_map = collections.OrderedDict({
        'word_embedding': "ernie.ernie.ernie_embedding_lookup.embedding_table",
        'pos_embedding': "ernie.ernie.ernie_embedding_postprocessor.full_position_embedding.embedding_table",
        'sent_embedding': "ernie.ernie.ernie_embedding_postprocessor.token_type_embedding.embedding_table",
        'pre_encoder_layer_norm_scale': 'ernie.ernie.ernie_embedding_postprocessor.layernorm.gamma',
        'pre_encoder_layer_norm_bias': 'ernie.ernie.ernie_embedding_postprocessor.layernorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.w_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.attention.query_layer.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.b_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.attention.query_layer.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.w_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.attention.key_layer.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.b_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.attention.key_layer.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.w_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.attention.value_layer.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.b_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.attention.value_layer.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.w_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.output.dense.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.b_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.output.dense.bias'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_scale'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.output.layernorm.gamma'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_bias'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.attention.output.layernorm.beta'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.w_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.intermediate.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.b_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.intermediate.bias'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.w_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.output.dense.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.b_0'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.output.dense.bias'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_scale'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.output.layernorm.gamma'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_bias'] = \
            f'ernie.ernie.ernie_encoder.layers.{i}.output.layernorm.beta'
    # add pooler
    weight_map.update(
        {
            'pooled_fc.w_0': 'ernie.ernie.dense.weight',
            'pooled_fc.b_0': 'ernie.ernie.dense.bias',
            'cls_out_w': 'ernie.dense_1.weight',
            'cls_out_b': 'ernie.dense_1.bias'
        }
    )
    return weight_map

def extract_and_convert(input_dir, output_dir):
    """extract weights and convert to mindspore"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config = json.load(open(os.path.join(input_dir, 'ernie_config.json'), 'rt', encoding='utf-8'))
    print('=' * 20 + 'save vocab file' + '=' * 20)
    shutil.copyfile(os.path.join(input_dir, 'vocab.txt'), os.path.join(output_dir, 'vocab.txt'))
    print('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = []
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    with fluid.dygraph.guard():
        paddle_paddle_params, _ = D.load_dygraph(os.path.join(input_dir, 'params'))
    for weight_name, weight_value in paddle_paddle_params.items():
        if weight_name not in weight_map.keys():
            continue
        #print(weight_name, weight_value.shape)
        if 'w_0' in weight_name \
            or 'post_att_layer_norm_scale' in weight_name \
            or 'post_ffn_layer_norm_scale' in weight_name \
            or 'cls_out_w' in weight_name:
            weight_value = weight_value.transpose()
        state_dict.append({'name': weight_map[weight_name], 'data': Tensor(weight_value)})
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    save_checkpoint(state_dict, os.path.join(output_dir, "ernie.ckpt"))

def run_convert():
    """run convert"""
    parser = argparse.ArgumentParser(description="run convert")
    parser.add_argument("--input_dir", type=str, default="", help="Pretrained model dir")
    parser.add_argument("--output_dir", type=str, default="", help="Converted model dir")
    args_opt = parser.parse_args()
    extract_and_convert(args_opt.input_dir, args_opt.output_dir)

if __name__ == '__main__':
    run_convert()
