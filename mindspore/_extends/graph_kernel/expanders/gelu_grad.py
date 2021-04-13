# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""generate json desc for gelugrad"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class GeLUGrad(Expander):
    """GeLUGrad expander"""
    CSVALUE = 0.044715
    CSVALUE_SQRT_TWO_DIV_PI = 0.7978845608028564  # np.sqrt(2/np.pi)
    CSVALUE_TRI = 0.134141  # CSVALUE * 3

    def _expand(self, graph_builder):
        # cal formula are:
        # gelu_grad of dy and x is dy * y'
        # y' is 0.5 * (1.0 + tanh(tanh_para)) + 0.5 * x * (1.0 - tanh(tanh_para) * tanh(para)) * mul_right
        # tanh_para is sqrt(2.0 / pi) * (x + 0.044715 * x * x * x)
        # mul_right is sqrt(2.0 / pi) * (1 + 3 * 0.044715 * x * x)

        input_dy, input_x, _ = self.inputs

        # create some const var
        const_csvalue = graph_builder.value(input_dy.dtype, self.CSVALUE)
        const_csvalue_sqrt_two_div_pi = graph_builder.value(input_dy.dtype, self.CSVALUE_SQRT_TWO_DIV_PI)
        const_csvalue_tri = graph_builder.value(input_dy.dtype, self.CSVALUE_TRI)
        const_one = graph_builder.value(input_dy.dtype, 1)
        const_half = graph_builder.value(input_dy.dtype, 0.5)

        # cal mul_right
        mul_double = graph_builder.emit('Mul', [input_x, input_x])
        mul_double_mul_tri = graph_builder.emit('Mul', [const_csvalue_tri, mul_double])
        mul_add_one = graph_builder.emit('Add', [const_one, mul_double_mul_tri])
        mul_right = graph_builder.emit('Mul', [const_csvalue_sqrt_two_div_pi, mul_add_one])

        # cal tanh_para
        mul_triple = graph_builder.emit('Mul', [input_x, mul_double])
        mul_triple_mul_csvalue = graph_builder.emit('Mul', [const_csvalue, mul_triple])
        mul_add_x = graph_builder.emit('Add', [input_x, mul_triple_mul_csvalue])
        tanh_para = graph_builder.emit('Mul', [const_csvalue_sqrt_two_div_pi, mul_add_x])

        # cal 0.5 * (1.0 + tanh(tahn_para))
        tanh_res = graph_builder.emit('Tanh', [tanh_para])
        tanh_res_add_one = graph_builder.emit('Add', [const_one, tanh_res])
        half_mul_tanh_res_add_one = graph_builder.emit('Mul', [const_half, tanh_res_add_one])

        # cal 0.5 * x * (1.0 - tanh(tanh_para) * tanh(tanh_para)) * mul_right
        tan_res_double = graph_builder.emit('Mul', [tanh_res, tanh_res])
        one_sub_tan_res_double = graph_builder.emit('Sub', [const_one, tan_res_double])
        half_mul_x = graph_builder.emit('Mul', [const_half, input_x])
        mul_tmp = graph_builder.emit('Mul', [half_mul_x, one_sub_tan_res_double])
        mul_final = graph_builder.emit('Mul', [mul_tmp, mul_right])

        # cal result
        result_tmp = graph_builder.emit('Add', [half_mul_tanh_res_add_one, mul_final])
        result = graph_builder.emit('Mul', [input_dy, result_tmp])

        return result
