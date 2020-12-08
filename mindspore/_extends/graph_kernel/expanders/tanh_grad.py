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
"""generate json desc for tanh_grad"""
from mindspore._extends.graph_kernel.model import model_builder as builder

ONE = 1.0


def expand_tanhgrad(expand_info):
    """TanhGrad expander"""

    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_y = graph_builder.tensor(input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        input_dy = graph_builder.tensor(input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        const_one = graph_builder.value(input_y.dtype, ONE, input_y.data_format)
        graph_scope.set_input(input_y, input_dy)

        # cal result
        double_y = graph_builder.emit('Mul', [input_y, input_y])
        one_sub_double_y = graph_builder.emit('Sub', [const_one, double_y])
        result = graph_builder.emit('Mul', [input_dy, one_sub_double_y])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
