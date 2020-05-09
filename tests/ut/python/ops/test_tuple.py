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
# ============================================================================
import mindspore.context as context
import functools
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore import context
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config
context.set_context(mode=context.GRAPH_MODE, save_graphs=True)


class TupleGraphNet(nn.Cell):
    def __init__(self):
        super(TupleGraphNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 3, pad_mode='same')
        self.conv2 = nn.Conv2d(3, 1, 7, pad_mode='same')
        self.conv3 = nn.Conv2d(3, 3, 3, pad_mode='same')
        self.layers = (self.conv1, self.conv2, self.conv3)

    def construct(self, x):
        return self.layers[0](x)


class NestTupleGraphNet(nn.Cell):
    def __init__(self):
        super(NestTupleGraphNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 3, pad_mode='same')
        self.conv2 = nn.Conv2d(3, 1, 7, pad_mode='same')
        self.conv3 = nn.Conv2d(3, 3, 3, pad_mode='same')
        self.layers = ((self.conv1, self.conv2),
                       (self.conv2, self.conv1, self.conv3))

    def construct(self, x):
        return self.layers[0][1](x)


test_case_ops = [
    ('TupleGraph', {
        'block': TupleGraphNet(),
        'desc_inputs': [Tensor(np.ones((3, 3, 24, 24)), mstype.float32)]}),
    ('NestTupleGraph', {
        'block': NestTupleGraphNet(),
        'desc_inputs': [Tensor(np.ones((3, 3, 24, 24)), mstype.float32)]}),
]

test_case_lists = [test_case_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm


@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case
