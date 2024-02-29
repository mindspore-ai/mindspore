# Copyright 2024 Huawei Technologies Co., Ltd
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

from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy

parameter_layout_dict = {
    'accu_grads.backbone.embedding.word_embedding.embedding_table':
        ([4, 4], [0, -1], [10000, 2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.attention.projection.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense1.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense2.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense3.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.output.mapping.weight':
        ([4, 4], [-1, 0], [2560, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.layernorm1.gamma':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.layernorm1.beta':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.layernorm2.gamma':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.attention.dense1.bias':
        ([4, 4], [0], [640], 0, True, ''),
}


def test_get_parameter_redundancy():
    """
    Feature: get parameter redundancy
    Description: get parameter for each rank
    Expectation: run success
    """
    param_redundancy_dict = get_parameter_redundancy(parameter_layout_dict)
    single_parameter = remove_param_redundancy(param_redundancy_dict)
    print(len(single_parameter))
