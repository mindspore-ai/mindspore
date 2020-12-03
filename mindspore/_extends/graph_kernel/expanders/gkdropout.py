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
"""generate json desc for GkDropOut"""
from mindspore._extends.graph_kernel.model import model_builder as builder


def expand_gkdropout(expand_info):
    """GkDropOut expander"""
    # get op info.
    input_desc = expand_info['input_desc'][0]
    maks_desc = expand_info['input_desc'][1]
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
        input_x = graph_builder.tensor(input_desc['shape'], input_desc['data_type'], input_desc['format'])
        input_mask = graph_builder.tensor(maks_desc['shape'], maks_desc['data_type'], maks_desc['format'])
        graph_scope.set_input(input_x, input_mask)
        keep_prob_v = graph_builder.value(input_x.dtype, keep_prob, "DefaultFormat")
        r_keep_prob = graph_builder.value(input_x.dtype, 1.0 / keep_prob, "DefaultFormat")

        if input_mask.dtype != input_x.dtype:
            input_mask = graph_builder.emit('Cast', [input_mask], attrs={'dst_type': input_x.dtype})
        mask = graph_builder.emit('LessEqual', [input_mask, keep_prob_v]) # output is bool type
        mask = graph_builder.emit('Cast', [mask], attrs={'dst_type': input_x.dtype})

        # compute result
        result = graph_builder.emit('Mul', [r_keep_prob, input_x])
        result = graph_builder.emit('Mul', [result, mask])
        # set graph output.
        graph_scope.set_output(result, mask)
    graph = graph_builder.get()[0]
    return graph
