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
"""generate json desc for LayerNorm"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_layernorm(expand_info):
    """LayerNorm expander"""
    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    input_desc_2 = expand_info['input_desc'][2]
    attrs = expand_info['attr']
    begin_norm_axis = None
    epsilon = None
    for item in attrs:
        if 'begin_norm_axis' in item:
            begin_norm_axis = item['begin_norm_axis']
        if 'epsilon' in item:
            epsilon = item['epsilon']
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        input_gamma = graph_builder.tensor(input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        input_beta = graph_builder.tensor(input_desc_2['shape'], input_desc_2['data_type'], input_desc_2['format'])

        # Calculate the scaling ratio of the average
        shape_x = input_desc_0['shape']
        if begin_norm_axis < 0:
            begin_norm_axis += len(shape_x)
        reduce_axis = ()
        for i, _ in enumerate(shape_x):
            if i > begin_norm_axis or i == begin_norm_axis:
                reduce_axis = reduce_axis + (i,)

        reduce_elts = 1.0
        for i in reduce_axis:
            reduce_elts *= shape_x[i]
        mean_cof = 1.0 / reduce_elts
        mean_cof_v = graph_builder.value(input_x.dtype, mean_cof, input_x.data_format)

        # Calculate mean
        mean_red = graph_builder.emit('ReduceSum', [input_x], attrs={'reduce_axis': reduce_axis, 'keep_dims': True})
        mean = graph_builder.emit('Mul', [mean_red, mean_cof_v])

        # Calculate variance
        variance_sub = graph_builder.emit('Sub', [input_x, mean])
        variance_mul = graph_builder.emit('Mul', [variance_sub, variance_sub])
        variance_red = graph_builder.emit('ReduceSum', [variance_mul],
                                          attrs={'reduce_axis': reduce_axis, 'keep_dims': True})
        variance = graph_builder.emit('Mul', [variance_red, mean_cof_v])

        # Calculate normalize
        normalize_sub = graph_builder.emit('Sub', [input_x, mean])
        epsilon_v = graph_builder.value(input_x.dtype, epsilon, input_x.data_format)
        normalize_add = graph_builder.emit('Add', [variance, epsilon_v])
        normlize_rsqrt = graph_builder.emit('Rsqrt', [normalize_add])
        normalize_mul = graph_builder.emit('Mul', [normalize_sub, normlize_rsqrt])

        # Calculate scale and translate
        scale_mul = graph_builder.emit('Mul', [input_gamma, normalize_mul])
        res = graph_builder.emit('Add', [scale_mul, input_beta])

        # set graph output.
        graph_scope.set_output(res, mean, variance)

    graph = graph_builder.get()[0]
    return graph
