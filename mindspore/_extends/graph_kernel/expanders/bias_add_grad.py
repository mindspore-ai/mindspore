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


def expand_biasaddgrad(expand_info):
    """BiasAddGrad expander"""
    # get op info.
    input_desc_0 = expand_info['input_desc'][0]
    graph_builder = builder.GraphBuilder()
    # generate a graph.
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_x = graph_builder.tensor(
            input_desc_0['shape'], input_desc_0['data_type'], input_desc_0['format'])
        graph_scope.set_input(input_x)
        reduce_axis = ()
        if input_x.data_format == 'NHWC':
            reduce_axis = (0, 1, 2)
        elif input_x.data_format == 'NCHW':
            reduce_axis = (0, 2, 3)
        # Default format shape's length maybe equal 2 to 4, so different shape's length reduce axis are differnet
        else:
            if len(input_x.shape) == 2:
                reduce_axis = (0,)
            elif len(input_x.shape) == 3:
                reduce_axis = (0, 1)
            else:
                reduce_axis = (0, 2, 3)
        result = graph_builder.emit('ReduceSum', [input_x], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})
        # set graph output.
        graph_scope.set_output(result)

    graph = graph_builder.get()[0]
    return graph
