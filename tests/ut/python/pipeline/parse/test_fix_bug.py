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
""" test_fix_bug """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as ms
from mindspore.common.api import _cell_graph_executor


class assignment1_Net(nn.Cell):
    """ assignment1_Net definition """

    def __init__(self, number):
        super().__init__()
        self.number = number
        self.relu = nn.ReLU()

    def construct(self, x):
        y = self.number
        for _ in [1, y]:
            x = self.relu(x)
        return x


class assignment2_Net(nn.Cell):
    """ assignment2_Net definition """

    def __init__(self, number):
        super().__init__()
        self.number = number
        self.relu = nn.ReLU()

    def construct(self, x):
        a, b = self.number
        for _ in [a, b]:
            x = self.relu(x)
        return x


def assignment_operator_base(number):
    """ assignment_operator_base """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    x = number
    if isinstance(x, int):
        net = assignment1_Net(x)
    else:
        net = assignment2_Net(x)
    _cell_graph_executor.compile(net, input_me)


def test_ME_assignment_operator_0010():
    """ test_ME_assignment_operator_0010 """
    assignment_operator_base(3)


def test_ME_assignment_operator_0020():
    """ test_ME_assignment_operator_0020 """
    assignment_operator_base((1, 3))


def test_parser_map_0002():
    class NetMap0002(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.hypermap = C.Map()

        def mul(self, x=2, y=4):
            return x * y

        def construct(self, x):
            if map(self.mul) == 8:
                x = self.relu(x)
            return x
    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)

    net = NetMap0002()
    with pytest.raises(TypeError):
        net(input_me_x)


def test_fix_expanddims_loss_scale():
    class ControlOneIfOneScaleOneScale(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.ExpandDims()

        def construct(self, x, y, data):
            if x > y:
                out = 1
            else:
                out = 2
            if x > y:
                out = self.op(data, out)
            else:
                out = self.op(data, out)
            return out
    net = ControlOneIfOneScaleOneScale()
    x = Tensor(1, ms.float32)
    y = Tensor(0, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input_data = np.random.randn(*input_shape).astype(np.float32)
    net = ControlOneIfOneScaleOneScale()
    net(x, y, Tensor(input_data))
