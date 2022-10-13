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
""" test view"""
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_view():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]])

        def construct(self):
            return self.value.view(-1)

    net = Net()
    net()


def test_view_initializer():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = init.initializer('normal', [2, 3], ms.float32)

        def construct(self):
            return self.value.view(-1)

    net = Net()
    net()


def test_view_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]])

        def construct(self):
            return self.value.view((3, 2))

    net = Net()
    net()


def test_view_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]])

        def construct(self):
            return self.value.view(3, 2)

    net = Net()
    net()


def test_view_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return x.view(-1)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6]]))


def test_view_parameter_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return x.view((3, 2))

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6]]))


def test_view_parameter_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return x.view(3, 2)

    net = Net()
    net(Tensor([[1, 2, 3], [4, 5, 6]]))


def test_view_shape_error():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]])

        def construct(self):
            return self.value.view()

    net = Net()
    with pytest.raises(ValueError) as ex:
        net()
    assert "The shape variable should not be empty" in str(ex.value)


def test_view_shape_error_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]])

        def construct(self):
            return self.value.view((2, 3), (4, 5))

    net = Net()
    with pytest.raises(ValueError) as ex:
        net()
    assert "Only one tuple is needed" in str(ex.value)
