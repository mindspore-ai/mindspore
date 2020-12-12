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


def _get_tile_output_shape(shape, multiples):
    """compute output shape of tile"""

    if multiples is None:
        return shape
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Input shape of Tile must be of type list or tuple")
    if not isinstance(multiples, (list, tuple)):
        raise TypeError("multiples of Tile must be of type list or tuple")

    shape = list(shape)
    multiples = list(multiples)
    diff_len = len(multiples) - len(shape)
    if diff_len < 0:
        raise ValueError("Dimensions of multiples{} < dimensions of input{} in Tile".format(multiples, shape))
    if diff_len > 0:
        for _ in range(diff_len):
            shape.insert(0, 1)

    shape_compatible = True
    output_shape = []
    input_reshape = []
    output_reshape = []
    for sh, mul in list(zip(shape, multiples)):
        dim = sh * mul
        output_shape.append(dim)
        if sh == 1 or mul == 1:
            input_reshape.append(sh)
            output_reshape.append(dim)
        else:
            shape_compatible = False
            input_reshape.append(1)
            input_reshape.append(sh)
            output_reshape.append(mul)
            output_reshape.append(sh)

    return output_shape, input_reshape, output_reshape, shape_compatible


def expand_tile(expand_info):
    """Tile expander"""

    # get op info.
    input_desc = expand_info['input_desc'][0]
    attrs = expand_info['attr']
    multiples = None
    for item in attrs:
        if 'multiples' in item:
            multiples = item['multiples']
    output_shape, _, _, shape_compatible = _get_tile_output_shape(input_desc['shape'], multiples)
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
