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
test nn.Tril()
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_tril():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        def construct(self):
            tril = nn.Tril()
            return tril(self.value, 0)

    net = Net()
    out = net()
    assert np.sum(out.asnumpy()) == 34


def test_tril_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        def construct(self):
            tril = nn.Tril()
            return tril(self.value, 1)

    net = Net()
    out = net()
    assert np.sum(out.asnumpy()) == 42


def test_tril_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        def construct(self):
            tril = nn.Tril()
            return tril(self.value, -1)

    net = Net()
    out = net()
    assert np.sum(out.asnumpy()) == 19


def test_tril_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            tril = nn.Tril()
            return tril(x, 0)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_tril_parameter_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            tril = nn.Tril()
            return tril(x, 1)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_tril_parameter_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            tril = nn.Tril()
            return tril(x, -1)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
