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
"""generate json desc for LogSoftmax"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_logsoftmax(expand_info):
    """LogSoftmax expander"""
    # get op info.
    input_desc = expand_info['input_desc'][0]
    attrs = expand_info['attr']
    axis = None
    for item in attrs:
        if 'axis' in item:
            axis = item['axis']
    graph_builder = builder.GraphBuilder()
    if isinstance(axis, int):
        axis = (axis,)
    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc['shape'], input_desc['data_type'], input_desc['format'])
        graph_scope.set_input(input_x)

        # cal logsoftmax.
        max_x = graph_builder.emit('ReduceMax', [input_x], attrs={'reduce_axis': axis, 'keep_dims': True})
        data_sub = graph_builder.emit('Sub', [input_x, max_x])
        data_exp = graph_builder.emit('Exp', [data_sub])
        data_expsum = graph_builder.emit('ReduceSum', [data_exp], attrs={'reduce_axis': axis, 'keep_dims': True})
        log_expsum = graph_builder.emit('Log', [data_expsum])
        result = graph_builder.emit('Sub', [data_sub, log_expsum])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
