# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test tensor properties in graph mode"""

import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_ndim():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor(np.random.random(
                (2, 3, 4, 5)), dtype=mstype.float32)

        def construct(self):
            return self.value.ndim

    net = Net()
    res = net()
    assert res == 4


def test_nbytes():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor(np.random.random(
                (2, 3, 4, 5)), dtype=mstype.float32)

        def construct(self):
            return self.value.nbytes

    net = Net()
    res = net()
    assert res == 480


def test_size():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor(np.random.random(
                (2, 3, 4, 5)), dtype=mstype.float32)

        def construct(self):
            return self.value.size

    net = Net()
    res = net()
    assert res == 120


def test_strides():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor(np.random.random(
                (2, 3, 4, 5)), dtype=mstype.float32)

        def construct(self):
            return self.value.strides

    net = Net()
    res = net()
    assert res == (240, 80, 20, 4)


def test_itemsize():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value1 = Tensor(np.random.random(
                (2, 3, 4, 5)), dtype=mstype.float64)
            self.value2 = Tensor(np.random.random(
                (2, 3, 4, 5)), dtype=mstype.int32)
            self.value3 = Tensor(np.random.random(
                (2, 3, 4, 5)), dtype=mstype.bool_)

        def construct(self):
            return (self.value1.itemsize, self.value2.itemsize, self.value3.itemsize)

    net = Net()
    res = net()
    assert res == (8, 4, 1)
