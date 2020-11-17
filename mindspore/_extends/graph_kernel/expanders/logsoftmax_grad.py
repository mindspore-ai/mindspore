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
"""generate json desc for LogSoftmaxGrad"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_logsoftmaxgrad(expand_info):
    """LogSoftmaxGrad expander"""
    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    attrs = expand_info['attr']
    axis = None
    for item in attrs:
        if 'axis' in item:
            axis = item['axis']
    graph_builder = builder.GraphBuilder()

    if isinstance(axis, int):
        axis = (axis,)
    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_logits = graph_builder.tensor(input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        input_dy = graph_builder.tensor(input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        graph_scope.set_input(input_logits, input_dy)

        # cal logsoftmaxgrad.
        softmax = graph_builder.emit('Exp', [input_logits])
        dy_sum = graph_builder.emit('ReduceSum', [input_dy], attrs={'reduce_axis': axis, 'keep_dims': True})
        mul_result = graph_builder.emit('Mul', [softmax, dy_sum])
        result = graph_builder.emit('Sub', [input_dy, mul_result])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
