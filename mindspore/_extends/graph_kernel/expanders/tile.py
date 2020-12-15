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
"""generate json desc for Tile"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_tile(expand_info):
    """Tile expander"""

    # get op info.
    input_desc = expand_info['input_desc'][0]
    attrs = expand_info['attr']
    multiples = None
    for item in attrs:
        if 'multiples' in item:
            multiples = item['multiples']
    output_shape, _, _, shape_compatible = builder.get_tile_output_shape(input_desc['shape'], multiples)
    graph_builder = builder.GraphBuilder()

    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(input_desc['shape'], input_desc['data_type'], input_desc['format'])
        # create op.
        if shape_compatible:
            result = graph_builder.emit('BroadcastTo', [input_x], attrs={'shape': output_shape})
        else:
            result = graph_builder.emit('Tile', [input_x], attrs={'multiples': multiples})
        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
