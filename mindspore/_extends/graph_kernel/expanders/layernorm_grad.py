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
"""generate json desc for LayerNormGrad"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_layernormgrad(expand_info):
    """LayerNormGrad expander"""
    # get op info.
    x_desc = expand_info['input_desc'][0]
    dy_desc = expand_info['input_desc'][1]
    var_desc = expand_info['input_desc'][2]
    mean_desc = expand_info['input_desc'][3]
    gamma_desc = expand_info['input_desc'][4]
    begin_norm_axis = None
    begin_params_axis = None
    epsilon = 1e-11
    for item in expand_info['attr']:
        if 'begin_norm_axis' in item:
            begin_norm_axis = item['begin_norm_axis']
        if 'begin_params_axis' in item:
            begin_params_axis = item['begin_params_axis']
        if 'epsilon' in item:
            epsilon = item['epsilon']

    shape_x = x_desc['shape']
    if begin_norm_axis < 0:
        begin_norm_axis += len(shape_x)
    if begin_params_axis < 0:
        begin_params_axis += len(shape_x)
    norm_axis = tuple(range(begin_norm_axis, len(shape_x)))
    param_axis = tuple(range(0, begin_params_axis))
    reduce_size = 1.0
    for i in norm_axis:
        reduce_size *= shape_x[i]

    graph_builder = builder.GraphBuilder()
    with graph_builder.graph_scope('main') as graph_scope:
        # create input tensors.
        x = graph_builder.tensor(x_desc['shape'], x_desc['data_type'], x_desc['format'])
        dy = graph_builder.tensor(dy_desc['shape'], dy_desc['data_type'], dy_desc['format'])
        variance = graph_builder.tensor(var_desc['shape'], var_desc['data_type'], var_desc['format'])
        mean = graph_builder.tensor(mean_desc['shape'], mean_desc['data_type'], mean_desc['format'])
        gamma = graph_builder.tensor(gamma_desc['shape'], gamma_desc['data_type'], gamma_desc['format'])
        graph_scope.set_input(x, dy, variance, mean, gamma)

        # set some constant val.
        eps = graph_builder.value(x.dtype, epsilon, x.data_format)
        const_one = graph_builder.value(x.dtype, 1.0, x.data_format)
        const_neg_half = graph_builder.value(x.dtype, -0.5, x.data_format)
        const_neg_two = graph_builder.value(x.dtype, -2.0, x.data_format)
        const_two = graph_builder.value(x.dtype, 2.0, x.data_format)
        const_neg_one = graph_builder.value(x.dtype, -1.0, x.data_format)
        mean_cof = graph_builder.value(x.dtype, (1.0 / reduce_size), x.data_format)

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

        # cal dx = dx1 + dx2 + dx3
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

        # set graph output.
        graph_scope.set_output(dx, dg, db)

    graph = graph_builder.get()[0]
    return graph
