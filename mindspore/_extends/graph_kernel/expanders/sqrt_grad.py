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
"""generate json desc for sqrtgrad"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_sqrtgrad(expand_info):
    """SqrtGrad expander"""
    # cal formula are:
    # sqrt_grad(x, dout) is dout / (2 * x)

    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        input_dout = graph_builder.tensor(input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        graph_scope.set_input(input_x, input_dout)

        # cal result
        const_two = graph_builder.value(input_x.dtype, 2, input_x.data_format)
        dividend = graph_builder.emit('Mul', [input_x, const_two])
        result = graph_builder.emit('RealDiv', [input_dout, dividend])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
