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
"""
test assign sub
"""
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore as ms

class AssignW(nn.Cell):
    def __init__(self):
        super(AssignW, self).__init__()
        self.assign = P.Assign()

    def construct(self, x, w):
        self.assign(x, w)
        return x


class AssignOp(nn.Cell):
    def __init__(self):
        super(AssignOp, self).__init__()
        self.b = Parameter(initializer('ones', [5]), name='b')


    def construct(self, w):
        self.b = w
        return w


def test_assign_by_operator():
    context.set_context(mode=context.GRAPH_MODE)
    net = AssignOp()
    net.to_float(ms.float16)
    input_data = Tensor(np.ones([5]).astype(np.float32))
    net(input_data)


class NetScatterNdUpdate(nn.Cell):
    def __init__(self):
        super(NetScatterNdUpdate, self).__init__()
        self.b = Parameter(initializer('ones', [5, 5]), name='b')
        self.scatter = P.ScatterNdUpdate()

    def construct(self, idx, x):
        return self.scatter(self.b, idx, x)


def test_scatter_nd_update():
    context.set_context(mode=context.GRAPH_MODE)
    net = NetScatterNdUpdate()
    x = Tensor(np.ones([5]).astype(np.float16))
    idx = Tensor(np.ones([1]).astype(np.int32))
    net(idx, x)
