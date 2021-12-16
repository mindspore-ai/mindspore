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
# ===========================================================================
"""generate json desc for LambApplyWeightAssign"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class LambApplyWeightAssign(Expander):
    """LambApplyWeightAssign expander"""

    def _expand(self, graph_builder):

        w_norm, g_norm, input_lr, update, input_param = self.inputs
        # ratio
        dtype = g_norm.dtype

        if dtype == "float32":
            data_min = graph_builder.value(dtype, 2**(-126))
        elif dtype == "float16":
            data_min = graph_builder.value(dtype, 2*(-24))
        else:
            raise ValueError("Only support float32 and float16, but input type is : {}!".format(dtype))

        const_zero = graph_builder.value(dtype, 0)
        const_one = graph_builder.value(dtype, 1)

        # w_norm >= 0, g_norm >= 0
        # ratio =  select(greater(w_norm, 0), select(greater(g_norm, 0), w_norm/g_norm, 1), 1)
        # cal ratio
        g_norm_greater_res = graph_builder.emit('Greater', [g_norm, const_zero])
        g_norm_res = graph_builder.emit('Cast', [g_norm_greater_res], attrs={'dst_type': dtype})

        g_norm = graph_builder.emit('Add', [g_norm, data_min])
        w_norm_g_norm = graph_builder.emit('RealDiv', [w_norm, g_norm])

        g_norm_value_1 = graph_builder.emit('Mul', [g_norm_res, w_norm_g_norm])
        # select
        g_norm_res_neg = graph_builder.emit('Neg', [g_norm_res])
        g_norm_res_f = graph_builder.emit('Add', [g_norm_res_neg, const_one])
        g_norm_value = graph_builder.emit('Add', [g_norm_value_1, g_norm_res_f])

        w_norm_greater_res = graph_builder.emit('Greater', [w_norm, const_zero])
        w_norm_res = graph_builder.emit('Cast', [w_norm_greater_res], attrs={'dst_type': dtype})

        w_norm_value_1 = graph_builder.emit('Mul', [w_norm_res, g_norm_value])
        # select
        w_norm_res_neg = graph_builder.emit('Neg', [w_norm_res])
        w_norm_res_f = graph_builder.emit('Add', [w_norm_res_neg, const_one])
        ratio = graph_builder.emit('Add', [w_norm_value_1, w_norm_res_f])

        # ratio * input_lr * update
        update_with_ir = graph_builder.emit('Mul', [update, input_lr])
        ratio_update_with_ir = graph_builder.emit('Mul', [update_with_ir, ratio])

        # input_param - ratio_update_with_ir
        next_param = graph_builder.emit('Sub', [input_param, ratio_update_with_ir])
        next_param = graph_builder.emit('Assign', [input_param, next_param])

        return [next_param]
