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
# ===========================================================================
"""generate json desc for minimum_grad"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_minimumgrad(expand_info):
    """MinimumGrad expander"""
    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    input_desc_2 = expand_info['input_desc'][2]
    attrs = expand_info['attr']
    grad_x = None
    grad_y = None
    for item in attrs:
        if 'grad_x' in item:
            grad_x = item['grad_x']
        if 'grad_y' in item:
            grad_y = item['grad_y']
    graph_builder = builder.GraphBuilder()
    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        input_y = graph_builder.tensor(input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        input_dout = graph_builder.tensor(input_desc_2['shape'], input_desc_2['data_type'], input_desc_2['format'])
        graph_scope.set_input(input_x, input_y, input_dout)
        x_dtype = input_x.dtype

        # cal result
        le_result = graph_builder.emit('LessEqual', [input_x, input_y])
        le_result = graph_builder.emit('Cast', [le_result], attrs={'dst_type': x_dtype})
        dx = graph_builder.emit('Mul', [le_result, input_dout])
        dy = graph_builder.emit('Sub', [input_dout, dx])

        # set graph output according to grad_x and grad_y
        if grad_x and grad_y:
            graph_scope.set_output(dx, dy)
        if grad_x and not grad_y:
            graph_scope.set_output(dx)
        if grad_y and not grad_x:
            graph_scope.set_output(dy)

    graph = graph_builder.get()[0]
    return graph
