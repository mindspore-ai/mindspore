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
""" test enumerate"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_enumerate_list_const():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self):
            index_sum = 0
            value_sum = 0
            for i, j in enumerate(self.value):
                index_sum += i
                value_sum += j
            return index_sum, value_sum

    net = Net()
    assert net() == (6, 110)


def test_enumerate_tuple_const():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = (11, 22, 33, 44)

        def construct(self):
            index_sum = 0
            value_sum = 0
            for i, j in enumerate(self.value):
                index_sum += i
                value_sum += j
            return index_sum, value_sum

    net = Net()
    assert net() == (6, 110)


def test_enumerate_list_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y, z):
            index_sum = 0
            value = [x, y, z]
            ret = ()
            for i, j in enumerate(value):
                index_sum += i
                ret += (j,)
            return index_sum, ret

    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    net = Net()
    net(x, x, x)


def test_enumerate_tuple_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y, z):
            index_sum = 0
            value = (x, y, z)
            ret = ()
            for i, j in enumerate(value):
                index_sum += i
                ret += (j,)
            return index_sum, ret

    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    net = Net()
    net(x, x, x)


def test_enumerate_tuple_const_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = (11, 22, 33, 44)

        def construct(self):
            index_sum = 0
            value_sum = 0
            for i in enumerate(self.value):
                index_sum += i[0]
                value_sum += i[1]
            return index_sum, value_sum

    net = Net()
    assert net() == (6, 110)


def test_enumerate_tuple_parameter_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y, z):
            index_sum = 0
            value = (x, y, z)
            ret = ()
            for i in enumerate(value):
                index_sum += i[0]
                ret += (i[1],)
            return index_sum, ret

    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    net = Net()
    net(x, x, x)


def test_enumerate_tuple_const_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = (11, 22, 33, 44)

        def construct(self):
            index_sum = 0
            value_sum = 0
            for i in enumerate(self.value, 1):
                index_sum += i[0]
                value_sum += i[1]
            return index_sum, value_sum

    net = Net()
    assert net() == (10, 110)


def test_enumerate_tuple_parameter_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y, z):
            index_sum = 0
            value = (x, y, z)
            ret = ()
            for i in enumerate(value, 2):
                index_sum += i[0]
                ret += (i[1],)
            return index_sum, ret

    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    net = Net()
    net(x, x, x)


def test_enumerate_first_input_type_error():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return enumerate(x)

    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    net = Net()
    with pytest.raises(TypeError) as ex:
        net(x)
    assert "For 'enumerate', the 'first input'" in str(ex.value)


def test_enumerate_start_type_error():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return enumerate(x, start=1.2)

    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    net = Net()
    with pytest.raises(TypeError) as ex:
        net((x, x))
    assert "For 'enumerate', the 'start'" in str(ex.value)
