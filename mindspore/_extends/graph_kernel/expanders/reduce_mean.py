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
"""generate json desc for reduce_mean"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_reducemean(expand_info):
    """ReduceMean expander"""

    # get op info.
    input_desc = expand_info['input_desc'][0]
    attrs = expand_info['attr']
    axis = None
    keep_dims = None
    for item in attrs:
        if 'axis' in item:
            axis = item['axis']
        if 'keep_dims' in item:
            keep_dims = item['keep_dims']
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc['shape'], input_desc['data_type'], input_desc['format'])
        x_shape = input_x.shape
        graph_scope.set_input(input_x)

        # cal reduce_mean, when axis = None, reduce axis are all
        all_shape = 1.0
        real_axis = []
        if not axis:
            for i, shape in enumerate(x_shape):
                real_axis.append(i)
                all_shape *= shape
        else:
            for idx in axis:
                all_shape *= x_shape[idx]

        all_shape_value = graph_builder.value(input_x.dtype, all_shape, input_x.data_format)

        if not axis:
            sum_x = graph_builder.emit('ReduceSum', [input_x], attrs={'reduce_axis': real_axis, 'keep_dims': keep_dims})
        else:
            sum_x = graph_builder.emit('ReduceSum', [input_x], attrs={'reduce_axis': axis, 'keep_dims': keep_dims})
        result = graph_builder.emit('RealDiv', [sum_x, all_shape_value])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
