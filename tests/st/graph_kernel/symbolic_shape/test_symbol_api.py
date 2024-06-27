# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor, Symbol
from mindspore.ops import functional as F
from mindspore.common.api import jit
from tests.mark_utils import arg_mark
import pytest


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_symbol_graphmode_setinputs():
    """
    Feature: Symbol
    Description: graphmode, set symbolic info with cell.set_inputs
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            return self.add(x, y)

    ms.set_context(mode=ms.GRAPH_MODE)

    s1 = Symbol(max=16, divisor=8)  # the value can be 8, 16
    s2 = Symbol(unique=True)
    x_dyn = Tensor(shape=[s1, s2, s2], dtype=ms.float32)
    y_dyn = Tensor(shape=[s1, s1, s2], dtype=ms.float32)
    net = Net()
    net.set_inputs(x_dyn, y_dyn)
    with pytest.raises(ValueError):
        x = Tensor(np.ones((32, 32, 32), np.float32))
        net(x, x)  # s1 > max

    with pytest.raises(ValueError):
        x = Tensor(np.ones((16, 8, 8), np.float32))
        y = Tensor(np.ones((16, 8, 1), np.float32))
        net(x, y)  # s2 is unique, but y.shape[2] != x.shape[2]

    with pytest.raises(ValueError):
        x = Tensor(np.ones((10, 8, 8), np.float32))
        net(x, x)  # s1.divisor = 8, but x.shape[0] == 10

    x = Tensor(np.ones((16, 8, 8), np.float32))
    assert net(x, x).shape == (16, 8, 8)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_symbol_pynativemode_setinputs():
    """
    Feature: Symbol
    Description: pynativemode, set symbolic info with cell.set_inputs
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = ops.Add()

        @jit
        def construct(self, x, y):
            return self.add(x, y)

    ms.set_context(mode=ms.PYNATIVE_MODE)

    s1 = Symbol(max=16, divisor=8)  # the value can be 8, 16
    s2 = Symbol(min=4, unique=True)
    x_dyn = Tensor(shape=[s1, s2], dtype=ms.float32)
    net = Net()
    net.set_inputs(x_dyn, x_dyn)

    with pytest.raises(ValueError):
        x = Tensor(np.ones((8, 1), np.float32))
        net(x, x)  # s2.min = 8, but y.shape[1] == 1

    x = Tensor(np.ones((16, 8), np.float32))
    assert net(x, x).shape == (16, 8)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_symbol_pynativemode_signature():
    """
    Feature: Symbol
    Description: pynativemode, set symbolic info with input_signature
    Expectation: success
    """
    s1 = Symbol(max=16, unique=True)
    s2 = Symbol(min=4, unique=True)
    x_dyn = Tensor(shape=[s1, s1], dtype=ms.float32)
    y_dyn = Tensor(shape=[s2, s2], dtype=ms.float32)
    @jit(input_signature=(x_dyn, y_dyn))
    def add_func(x, y):
        return F.tensor_add(x, y)

    ms.set_context(mode=ms.PYNATIVE_MODE)

    with pytest.raises(ValueError):
        x = Tensor(np.ones((1, 1), np.float32))
        y = Tensor(np.ones((4, 8), np.float32))
        add_func(x, y)  # s2 is unique, but y.shape[0] != y.shape[1]

    x = Tensor(np.ones((1, 1), np.float32))
    y = Tensor(np.ones((4, 4), np.float32))
    assert add_func(x, y).shape == (4, 4)
