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
"""generate json desc for fused_adam"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_fusedadam(expand_info):
    """FusedAdma expander"""
    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    input_desc_2 = expand_info['input_desc'][2]
    input_desc_3 = expand_info['input_desc'][3]
    input_desc_4 = expand_info['input_desc'][4]
    input_desc_5 = expand_info['input_desc'][5]
    input_desc_6 = expand_info['input_desc'][6]
    input_desc_7 = expand_info['input_desc'][7]
    input_desc_8 = expand_info['input_desc'][8]
    input_desc_9 = expand_info['input_desc'][9]
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        beta_1 = graph_builder.tensor(input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        one_sub_beta_1 = graph_builder.tensor(input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        beta_2 = graph_builder.tensor(input_desc_2['shape'], input_desc_2['data_type'], input_desc_2['format'])
        one_sub_beta_2 = graph_builder.tensor(input_desc_3['shape'], input_desc_3['data_type'], input_desc_3['format'])
        eps = graph_builder.tensor(input_desc_4['shape'], input_desc_4['data_type'], input_desc_4['format'])
        lr = graph_builder.tensor(input_desc_5['shape'], input_desc_5['data_type'], input_desc_5['format'])
        param = graph_builder.tensor(input_desc_6['shape'], input_desc_6['data_type'], input_desc_6['format'])
        m = graph_builder.tensor(input_desc_7['shape'], input_desc_7['data_type'], input_desc_7['format'])
        v = graph_builder.tensor(input_desc_8['shape'], input_desc_8['data_type'], input_desc_8['format'])
        gradient = graph_builder.tensor(input_desc_9['shape'], input_desc_9['data_type'], input_desc_9['format'])
        graph_scope.set_input(beta_1, one_sub_beta_1, beta_2, one_sub_beta_2, eps, lr, param, m, v, gradient)

        # compute result
        beta_1_mul_m = graph_builder.emit('Mul', [beta_1, m])
        one_sub_beta_1_mul_grad = graph_builder.emit('Mul', [one_sub_beta_1, gradient])
        next_m = graph_builder.emit('Add', [beta_1_mul_m, one_sub_beta_1_mul_grad])
        beta_2_mul_v = graph_builder.emit('Mul', [beta_2, v])
        grad_square = graph_builder.emit('Mul', [gradient, gradient])
        one_sub_beta_2_mul_grad_square = graph_builder.emit('Mul', [one_sub_beta_2, grad_square])
        next_v = graph_builder.emit('Add', [beta_2_mul_v, one_sub_beta_2_mul_grad_square])
        sqrt_next_v = graph_builder.emit('Sqrt', [next_v])
        sqrt_next_v_add_eps = graph_builder.emit('Add', [sqrt_next_v, eps])
        update = graph_builder.emit('RealDiv', [next_m, sqrt_next_v_add_eps])
        update_with_lr = graph_builder.emit('Mul', [lr, update])
        next_para = graph_builder.emit('Sub', [param, update_with_lr])

        param_result = graph_builder.emit('InplaceAssign', [param, next_para, next_para], attrs={'fake_output': True})
        param_result = graph_builder.emit('InplaceAssign', [m, next_m, param_result], attrs={'fake_output': True})
        param_result = graph_builder.emit('InplaceAssign', [v, next_v, param_result], attrs={'fake_output': True})

        # set graph output.
        graph_scope.set_output(param_result)

    graph = graph_builder.get()[0]
    return graph
