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
"""generate json desc for gelu"""
from mindspore._extends.graph_kernel.model import model_builder as builder

CSVALUE = 0.044715
CSVALUE_SQRT_TWO_DIV_PI = 0.7978845608028564  # np.sqrt(2/np.pi)
ONE = 1.0
HALF = 0.5


def expand_gelu(expand_info):
    """GeLU expander"""
    # cal formula are:
    # gelu(x) is 0.5 * x * (1.0 + tanh(y))
    # y is sqrt(2.0 / pi) * (x + 0.044715 * x * x * x)

    # get op info.
    input_desc = expand_info['input_desc'][0]
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc['shape'], input_desc['data_type'], input_desc['format'])
        graph_scope.set_input(input_x)

        # cal y
        mul_0 = graph_builder.emit('Mul', [input_x, input_x])
        pow_0 = graph_builder.emit('Mul', [mul_0, input_x])
        const_csvalue = graph_builder.value(pow_0.dtype, CSVALUE, input_desc['format'])
        mul_1 = graph_builder.emit('Mul', [pow_0, const_csvalue])
        tanh_res = graph_builder.emit('Add', [input_x, mul_1])
        const_csvalue_sqrt_two_div_pi = graph_builder.value(
            tanh_res.dtype, CSVALUE_SQRT_TWO_DIV_PI, input_desc['format'])
        y = graph_builder.emit('Mul', [tanh_res, const_csvalue_sqrt_two_div_pi])

        # cal gelu(x)
        tanh_y = graph_builder.emit('Tanh', [y])
        const_one = graph_builder.value(tanh_y.dtype, ONE, input_desc['format'])
        const_half = graph_builder.value(tanh_y.dtype, HALF, input_desc['format'])
        tanh_y_add_one = graph_builder.emit('Add', [tanh_y, const_one])
        mul_x = graph_builder.emit('Mul', [input_x, tanh_y_add_one])
        result = graph_builder.emit('Mul', [const_half, mul_x])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
