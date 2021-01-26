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
"""generate json desc for bias_add"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_biasadd(expand_info):
    """BiasAdd expander"""

    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    input_desc_1 = expand_info['input_desc'][1]
    graph_builder = builder.GraphBuilder()
    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(
            input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        input_y = graph_builder.tensor(
            input_desc_1['shape'], input_desc_1['data_type'], input_desc_1['format'])
        graph_scope.set_input(input_x, input_y)
        if input_x.data_format == "NCHW":
            input_y_expand = graph_builder.emit(
                'ExpandDims', [input_y], attrs={'axis': 1})
            input_y_expand = graph_builder.emit(
                'ExpandDims', [input_y_expand], attrs={'axis': 2})
            result = graph_builder.emit('Add', [input_x, input_y_expand])
        elif input_x.data_format == "DefaultFormat":
            if len(input_x.shape) == 2:
                result = graph_builder.emit('Add', [input_x, input_y])
            elif len(input_x.shape) == 3:
                input_y_expand = graph_builder.emit(
                    'ExpandDims', [input_y], attrs={'axis': 1})
                result = graph_builder.emit(
                    'Add', [input_x, input_y_expand])
            else:
                input_y_expand = graph_builder.emit(
                    'ExpandDims', [input_y], attrs={'axis': 1})
                input_y_expand = graph_builder.emit(
                    'ExpandDims', [input_y_expand], attrs={'axis': 2})
                result = graph_builder.emit(
                    'Add', [input_x, input_y_expand])
        else:
            result = graph_builder.emit('Add', [input_x, input_y])

        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
