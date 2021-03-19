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
"""generate json desc for LambApplyOptimizerAssign"""
from ._utils import Expander, ExpanderInfoValidator as VLD

@VLD.check_all_formats_same
class LambApplyOptimizerAssign(Expander):
    """LambApplyOptimizerAssign expander"""

    def _expand(self, graph_builder):

        [grad, inputv, inputm, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon, steps,
         do_use_weight, weight_decay_rate] = self.inputs

        # next_v
        square_grad = graph_builder.emit('Mul', [grad, grad])
        mul_3_result = graph_builder.emit('Mul', [square_grad, one_minus_beta_2])
        mul_2_result = graph_builder.emit('Mul', [inputv, beta_2])
        next_v = graph_builder.emit('Add', [mul_2_result, mul_3_result])

        # next_m
        mul_0_result = graph_builder.emit('Mul', [inputm, beta_1])
        mul_1_result = graph_builder.emit('Mul', [grad, one_minus_beta_1])
        next_m = graph_builder.emit('Add', [mul_0_result, mul_1_result])

        shape = next_m.shape
        const_one = graph_builder.value(beta_2.dtype, 1)

        beta_1_tensor = graph_builder.emit('BroadcastTo', [beta_1], attrs={'shape': shape})
        beta_2_tensor = graph_builder.emit('BroadcastTo', [beta_2], attrs={'shape': shape})


        # pow
        beta_1_log = graph_builder.emit('Log', [beta_1_tensor])
        mul_res_1 = graph_builder.emit('Mul', [beta_1_log, steps])
        beta_1_steps = graph_builder.emit('Exp', [mul_res_1])

        neg_beta_1_step = graph_builder.emit('Neg', [beta_1_steps])
        beta1_correction = graph_builder.emit('Add', [neg_beta_1_step, const_one])

        next_m_unbiased = graph_builder.emit('RealDiv', [next_m, beta1_correction])

        # pow
        beta_2_log = graph_builder.emit('Log', [beta_2_tensor])
        mul_res_2 = graph_builder.emit('Mul', [beta_2_log, steps])
        beta_2_steps = graph_builder.emit('Exp', [mul_res_2])

        neg_beta_2_step = graph_builder.emit('Neg', [beta_2_steps])
        beta2_correction = graph_builder.emit('Add', [neg_beta_2_step, const_one])

        next_v_unbiased = graph_builder.emit('RealDiv', [next_v, beta2_correction])
        # update
        sqrt_next_v = graph_builder.emit('Sqrt', [next_v_unbiased])

        add_2_result = graph_builder.emit('Add', [sqrt_next_v, epsilon])
        update = graph_builder.emit('RealDiv', [next_m_unbiased, add_2_result])
        # update do_use_weight_decay
        do_use_weight_mul = graph_builder.emit('Mul', [input_param, weight_decay_rate])
        do_use_weight_decay = graph_builder.emit('Mul', [do_use_weight_mul, do_use_weight])
        update = graph_builder.emit('Add', [do_use_weight_decay, update])

        res = [update, next_v, next_m]

        return res
