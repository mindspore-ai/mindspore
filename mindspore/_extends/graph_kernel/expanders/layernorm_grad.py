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
        processor = self.processor
        begin_norm_axis = self.attrs['begin_norm_axis']
        begin_params_axis = self.attrs['begin_params_axis']
        epsilon = self.attrs['epsilon'] if 'epsilon' in self.attrs else 1e-12

        ori_dtype = x.dtype
        if processor == 'aicore' and ori_dtype == 'float16':
            x = graph_builder.emit('Cast', [x], attrs={'dst_type': 'float32'})
            dy = graph_builder.emit('Cast', [dy], attrs={'dst_type': 'float32'})
            variance = graph_builder.emit('Cast', [variance], attrs={'dst_type': 'float32'})
            mean = graph_builder.emit('Cast', [mean], attrs={'dst_type': 'float32'})
            gamma = graph_builder.emit('Cast', [gamma], attrs={'dst_type': 'float32'})

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
        const_neg_half = graph_builder.value(x.dtype, -0.5)
        const_neg_two = graph_builder.value(x.dtype, -2.0)
        const_two = graph_builder.value(x.dtype, 2.0)
        const_neg_one = graph_builder.value(x.dtype, -1.0)
        mean_cof = graph_builder.value(x.dtype, (1.0 / reduce_size))

        # cal dg db
        var_eps = graph_builder.emit('Add', [variance, eps])
        var_eps_log = graph_builder.emit('Log', [var_eps])
        var_eps_mul = graph_builder.emit('Mul', [var_eps_log, const_neg_half])
        rsqrt_var_eps = graph_builder.emit('Exp', [var_eps_mul])

        x_sub_mean = graph_builder.emit('Sub', [x, mean])

        x_sub_mean_mul_rsqrt_var_eps = graph_builder.emit('Mul', [rsqrt_var_eps, x_sub_mean])
        dg_mul = graph_builder.emit('Mul', [dy, x_sub_mean_mul_rsqrt_var_eps])
        dg = graph_builder.emit('ReduceSum', [dg_mul], attrs={'reduce_axis': param_axis, 'keep_dims': False})
        db = graph_builder.emit('ReduceSum', [dy], attrs={'reduce_axis': param_axis, 'keep_dims': False})

        # pd_var
        tmp_var_eps = graph_builder.emit('Mul', [rsqrt_var_eps, rsqrt_var_eps])
        r_tmp_var_eps = graph_builder.emit('Mul', [rsqrt_var_eps, tmp_var_eps])

        dy_mul_gamma = graph_builder.emit('Mul', [dy, gamma])
        tmp_mul = graph_builder.emit('Mul', [dy_mul_gamma, x_sub_mean])
        padvar_mul1 = graph_builder.emit('ReduceSum', [tmp_mul], attrs={'reduce_axis': norm_axis, 'keep_dims': True})
        padvar_mul3 = graph_builder.emit('Mul', [padvar_mul1, r_tmp_var_eps])
        pd_var = graph_builder.emit('Mul', [padvar_mul3, const_neg_half])

        # pd_mean
        pdmean1_sum = graph_builder.emit('ReduceSum', [dy_mul_gamma],
                                         attrs={'reduce_axis': norm_axis, 'keep_dims': True})
        neg_rsqrt_var_eps = graph_builder.emit('Mul', [rsqrt_var_eps, const_neg_one])
        pd_mean_1 = graph_builder.emit('Mul', [neg_rsqrt_var_eps, pdmean1_sum])

        pdmean2_mul1 = graph_builder.emit('Mul', [const_neg_two, x_sub_mean])
        pdmean2_sum = graph_builder.emit('ReduceSum', [pdmean2_mul1],
                                         attrs={'reduce_axis': norm_axis, 'keep_dims': True})
        pdmean2_mul3 = graph_builder.emit('Mul', [pdmean2_sum, mean_cof])
        pd_mean_2 = graph_builder.emit('Mul', [pdmean2_mul3, pd_var])

        pd_mean = graph_builder.emit('Add', [pd_mean_1, pd_mean_2])

        # cal dx
        pd_x_1 = graph_builder.emit('Mul', [dy_mul_gamma, rsqrt_var_eps])

        pdx2_mul = graph_builder.emit('Mul', [pd_var, x_sub_mean])
        pdx2_mul_two = graph_builder.emit('Mul', [pdx2_mul, const_two])
        pd_x_2 = graph_builder.emit('Mul', [pdx2_mul_two, mean_cof])

        pd_x_3 = graph_builder.emit('Mul', [pd_mean, mean_cof])

        dx_tmp = graph_builder.emit('Add', [pd_x_1, pd_x_2])
        dx = graph_builder.emit('Add', [dx_tmp, pd_x_3])

        if processor == 'aicore' and ori_dtype == 'float16':
            dx = graph_builder.emit('Cast', [dx], attrs={'dst_type': 'float16'})
            dg = graph_builder.emit('Cast', [dg], attrs={'dst_type': 'float16'})
            db = graph_builder.emit('Cast', [db], attrs={'dst_type': 'float16'})
        return dx, dg, db
