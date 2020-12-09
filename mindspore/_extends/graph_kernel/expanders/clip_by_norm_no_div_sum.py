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
"""generate json desc for ClipByNormNoDivSum"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_clipbynormnodivsum(expand_info):
    """ClipByNormNoDivSum expander"""

    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    input_desc_2 = expand_info['input_desc'][2]
    input_desc_3 = expand_info['input_desc'][3]
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x0 = graph_builder.tensor(input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        input_x1 = graph_builder.tensor(input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        input_x2 = graph_builder.tensor(input_desc_2['shape'], input_desc_2['data_type'], input_desc_2['format'])
        input_x3 = graph_builder.tensor(input_desc_3['shape'], input_desc_3['data_type'], input_desc_3['format'])
        graph_scope.set_input(input_x0, input_x1, input_x2, input_x3)

        # cal result
        greater_res = graph_builder.emit('Greater', [input_x0, input_x1], attrs={'fusion': 'SelectGT_000'})
        select_res0 = graph_builder.emit('Select', [greater_res, input_x0, input_x2],
                                         attrs={'fusion': 'SelectGT_000_end'})
        sqrt_res = graph_builder.emit('Sqrt', [select_res0])
        select_res1 = graph_builder.emit('Select', [greater_res, sqrt_res, input_x0],
                                         attrs={'fusion': 'SelectGT_000_end'})
        result = graph_builder.emit('Maximum', [select_res1, input_x3])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
