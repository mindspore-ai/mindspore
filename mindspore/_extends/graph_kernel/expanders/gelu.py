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
CSVALUE_A = 1.5957691  # 2*np.sqrt(2/np.pi)


def expand_gelu(expand_info):
    """Gelu expander"""

    # get op info.
    input_desc = expand_info['input_desc'][0]
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc['shape'], input_desc['data_type'], input_desc['format'])
        dtype = input_x.dtype
        if dtype == 'float16':
            input_x = graph_builder.emit('Cast', [input_x], attrs={'dst_type': 'float32'})

        # cal tanh.
        mul_0 = graph_builder.emit('Mul', [input_x, input_x])
        pow_0 = graph_builder.emit('Mul', [mul_0, input_x])
        const_csvalue = graph_builder.value(pow_0.dtype, CSVALUE, input_desc['format'])
        mul_1 = graph_builder.emit('Mul', [pow_0, const_csvalue])
        tanh_res = graph_builder.emit('TensorAdd', [input_x, mul_1])

        const_csvalue_a = graph_builder.value(tanh_res.dtype, CSVALUE_A, input_desc['format'])
        mul_0 = graph_builder.emit('Mul', [tanh_res, const_csvalue_a])

        const_zero = graph_builder.value(mul_0.dtype, 0.0, input_desc['format'])
        mul_0_min = graph_builder.emit('Minimum', [mul_0, const_zero])
        right_mul = graph_builder.emit('Exp', [mul_0_min])

        mul_0_abs = graph_builder.emit('Abs', [mul_0])
        const_neg_one = graph_builder.value(mul_0_abs.dtype, -1.0, input_desc['format'])
        mul_0_abs_neg = graph_builder.emit('Mul', [mul_0_abs, const_neg_one])

        mul_0_abs_neg_exp = graph_builder.emit('Exp', [mul_0_abs_neg])

        const_one = graph_builder.value(mul_0_abs_neg_exp.dtype, 1.0, input_desc['format'])
        mul_0_abs_neg_exp_add = graph_builder.emit('TensorAdd', [mul_0_abs_neg_exp, const_one])
        left_mul = graph_builder.emit('RealDiv', [input_x, mul_0_abs_neg_exp_add])

        result = graph_builder.emit('Mul', [left_mul, right_mul])
        if dtype == 'float16':
            result = graph_builder.emit('Cast', [result], attrs={'dst_type': 'float16'})
        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
