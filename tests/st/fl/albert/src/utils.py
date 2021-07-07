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

import copy
from mindspore.common.initializer import initializer


def average_weights(para_list):
    global_parameter = {}
    length = len(para_list)
    for para in para_list:
        for name in para:
            if name in global_parameter:
                global_parameter[name] += para[name] / length
            else:
                global_parameter[name] = para[name] / length
    return global_parameter


def save_params(network, param_dict=None):
    if param_dict is None:
        return {param.name: copy.deepcopy(param) for param in network.trainable_params()
                if 'learning_rate' not in param.name and 'adam' not in param.name}
    for param in network.trainable_params():
        if param.name in param_dict:
            param_dict[param.name] = copy.deepcopy(param)
    return None


def restore_params(network, param_dict, init_adam=True):
    for param in network.trainable_params():
        if 'learning_rate' in param.name:
            continue
        param.init_data()
        if init_adam:
            if 'adam' in param.name:
                param.set_data(initializer('zeros', shape=param.shape, dtype=param.dtype))
            elif param.name in param_dict:
                param.set_data(param_dict[param.name])
        else:
            if param.name in param_dict:
                param.set_data(param_dict[param.name])


def get_worker_upload_list():
    return [
        'albert.encoder.embedding_hidden_mapping_in.weight',
        'albert.encoder.embedding_hidden_mapping_in.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.query.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.query.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.key.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.key.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.value.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.value.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.dense.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.dense.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.layernorm.gamma',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.layernorm.beta',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.gamma',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.beta',
        'albert.pooler.weight',
        'albert.pooler.bias',
        'classifier.weight',
        'classifier.bias']

def upload_to_server(network, worker_upload_list):
    for param in network.trainable_params():
        if param.name in worker_upload_list:
            param.set_param_fl(push_to_server=True)

def get_worker_download_list():
    return [
        'albert.encoder.embedding_hidden_mapping_in.weight',
        'albert.encoder.embedding_hidden_mapping_in.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.query.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.query.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.key.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.key.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.value.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.value.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.dense.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.dense.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.layernorm.gamma',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.layernorm.beta',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.gamma',
        'albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.beta'
    ]

def download_from_server(network, worker_download_list):
    for param in network.trainable_params():
        if param.name in worker_download_list:
            param.set_param_fl(pull_from_server=True)

def get_freeze_list():
    return [
        'albert.word_embeddings.embedding_table',
        'albert.embedding_postprocessor.embedding_table',
        'albert.embedding_postprocessor.full_position_embeddings',
        'albert.embedding_postprocessor.layernorm.gamma',
        'albert.embedding_postprocessor.layernorm.beta'
    ]

def freeze(network, freeze_list):
    for param in network.trainable_params():
        if param.name in freeze_list:
            param.requires_grad = False
