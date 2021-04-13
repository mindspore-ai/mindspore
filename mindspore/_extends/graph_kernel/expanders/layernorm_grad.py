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
"""generate json desc for LayerNormGrad"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('begin_norm_axis', 'begin_params_axis')
class LayerNormGrad(Expander):
    """LayerNormGrad expander"""

    def _expand(self, graph_builder):
        x, dy, variance, mean, gamma = self.inputs
        begin_norm_axis = self.attrs['begin_norm_axis']
        begin_params_axis = self.attrs['begin_params_axis']
        epsilon = self.attrs['epsilon'] if 'epsilon' in self.attrs else 1e-11

        if begin_norm_axis < 0:
            begin_norm_axis += len(x.shape)
        if begin_params_axis < 0:
            begin_params_axis += len(x.shape)
        norm_axis = tuple(range(begin_norm_axis, len(x.shape)))
        param_axis = tuple(range(0, begin_params_axis))
        reduce_size = 1.0
        for i in norm_axis:
            reduce_size *= x.shape[i]

        # set some constant val.
        eps = graph_builder.value(x.dtype, epsilon)
        const_one = graph_builder.value(x.dtype, 1.0)
        const_neg_half = graph_builder.value(x.dtype, -0.5)
        const_neg_two = graph_builder.value(x.dtype, -2.0)
        const_two = graph_builder.value(x.dtype, 2.0)
        const_neg_one = graph_builder.value(x.dtype, -1.0)
        mean_cof = graph_builder.value(x.dtype, (1.0 / reduce_size))

        # cal dg db
        var_eps = graph_builder.emit('Add', [variance, eps])
        sqrt_var_eps = graph_builder.emit('Sqrt', [var_eps])
        rsqrt_var_eps = graph_builder.emit('RealDiv', [const_one, sqrt_var_eps])
        x_sub_mean = graph_builder.emit('Sub', [x, mean])
        x_sub_mean_mul_rsqrt_var_eps = graph_builder.emit('Mul', [rsqrt_var_eps, x_sub_mean])
        dg_mul = graph_builder.emit('Mul', [dy, x_sub_mean_mul_rsqrt_var_eps])
        dg = graph_builder.emit('ReduceSum', [dg_mul], attrs={'reduce_axis': param_axis, 'keep_dims': False})
        db = graph_builder.emit('ReduceSum', [dy], attrs={'reduce_axis': param_axis, 'keep_dims': False})

        # cal sum_1
        tmp_var_eps = graph_builder.emit('Mul', [sqrt_var_eps, var_eps])
        r_tmp_var_eps = graph_builder.emit('RealDiv', [const_one, tmp_var_eps])
        x_sub_mean_mul_r_tmp_var_eps = graph_builder.emit('Mul', [x_sub_mean, r_tmp_var_eps])
        dy_mul_gamma = graph_builder.emit('Mul', [dy, gamma])
        tmp_mul = graph_builder.emit('Mul', [dy_mul_gamma, x_sub_mean_mul_r_tmp_var_eps])
        sum_1_mul = graph_builder.emit('Mul', [const_neg_half, tmp_mul])
        sum_1 = graph_builder.emit('ReduceSum', [sum_1_mul], attrs={'reduce_axis': norm_axis, 'keep_dims': True})

        # cal sum_2
        sum_2 = graph_builder.emit('ReduceSum', [dy_mul_gamma], attrs={'reduce_axis': norm_axis, 'keep_dims': True})

        # cal sum_3
        sum_3_mul = graph_builder.emit('Mul', [const_neg_two, x_sub_mean])
        sum_3 = graph_builder.emit('ReduceSum', [sum_3_mul], attrs={'reduce_axis': norm_axis, 'keep_dims': True})

        # cal dx, which is dx1 + dx2 + dx3
        dx_1 = graph_builder.emit('Mul', [dy_mul_gamma, rsqrt_var_eps])
        sum_1_mul_two = graph_builder.emit('Mul', [sum_1, const_two])
        sum_1_mul_two_tmp = graph_builder.emit('Mul', [sum_1_mul_two, mean_cof])
        dx_2 = graph_builder.emit('Mul', [sum_1_mul_two_tmp, x_sub_mean])
        neg_rsqrt_var_eps = graph_builder.emit('Mul', [const_neg_one, rsqrt_var_eps])
        neg_rsqrt_var_eps_mul_sum_2 = graph_builder.emit('Mul', [neg_rsqrt_var_eps, sum_2])
        sum_1_mul_sum_3 = graph_builder.emit('Mul', [sum_1, sum_3])
        mean_cof_mul_sum_1_mul_sum_3 = graph_builder.emit('Mul', [mean_cof, sum_1_mul_sum_3])
        add_tmp = graph_builder.emit('Add', [neg_rsqrt_var_eps_mul_sum_2, mean_cof_mul_sum_1_mul_sum_3])
        dx_3 = graph_builder.emit('Mul', [add_tmp, mean_cof])
        dx_tmp = graph_builder.emit('Add', [dx_1, dx_2])
        dx = graph_builder.emit('Add', [dx_tmp, dx_3])

        return dx, dg, db
