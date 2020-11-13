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
"""generate json desc for DropoutGrad"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_dropoutgrad(expand_info):
    """DropoutGrad expander"""
    # get op info.
    dy_desc = expand_info['input_desc'][0]
    mask_desc = expand_info['input_desc'][1]
    keep_prob = None
    for attr in expand_info['attr']:
        if 'keep_prob' in attr:
            keep_prob = attr['keep_prob']
    if keep_prob is None:
        raise RuntimeError("keep_prob does not exist in attrs.")
    # generate a graph.
    graph_builder = builder.GraphBuilder()
    with graph_builder.graph_scope('main') as graph_scope:
        # create tensor input.
        input_dy = graph_builder.tensor(dy_desc['shape'], dy_desc['data_type'], dy_desc['format'])
        input_mask = graph_builder.tensor(mask_desc['shape'], mask_desc['data_type'], mask_desc['format'])
        graph_scope.set_input(input_dy, input_mask)
        r_keep_prob = graph_builder.value(input_dy.dtype, 1.0 / keep_prob, "DefaultFormat")
        # create op.
        result = graph_builder.emit('Mul', [input_dy, r_keep_prob])
        result = graph_builder.emit('Mul', [result, input_mask])
        # set graph output.
        graph_scope.set_output(result)
    graph = graph_builder.get()[0]
    return graph
