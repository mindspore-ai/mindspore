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
""" Test Interpolate """
import pytest

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_resizebilinear():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mstype.float32)

        def construct(self):
            interpolate = nn.ResizeBilinear()
            return interpolate(self.value, size=(5, 5))

    net = Net()
    net()


def test_resizebilinear_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mstype.float32)

        def construct(self):
            interpolate = nn.ResizeBilinear()
            return interpolate(self.value, scale_factor=2)

    net = Net()
    net()


def test_resizebilinear_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            interpolate = nn.ResizeBilinear()
            return interpolate(x, size=(5, 5))

    net = Net()
    net(Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mstype.float32))


def test_resizebilinear_parameter_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            interpolate = nn.ResizeBilinear()
            return interpolate(x, scale_factor=2)

    net = Net()
    net(Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mstype.float32))


def test_resizebilinear_error():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mstype.float32)

        def construct(self):
            interpolate = nn.ResizeBilinear()
            return interpolate(self.value)

    net = Net()
    with pytest.raises(ValueError) as ex:
        net()
    assert "'size' and 'scale' both none" in str(ex.value)


def test_resizebilinear_error_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mstype.float32)

        def construct(self):
            interpolate = nn.ResizeBilinear()
            return interpolate(self.value, size=(5, 5), scale_factor=2)

    net = Net()
    with pytest.raises(ValueError) as ex:
        net()
    assert "'size' and 'scale' both not none" in str(ex.value)
