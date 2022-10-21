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
import operator
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import jit, context, Tensor

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


def test_enumerate_tensor_const():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor(np.arange(2 * 3).reshape(2, 3))

        def construct(self):
            return enumerate(self.value)

    net = Net()
    net()


def test_enumerate_list_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            index_sum = 0
            value = [x, y]
            ret = ()
            for i, j in enumerate(value):
                index_sum += i
                ret += (j,)
            return index_sum, ret

    x = Tensor(np.arange(4))
    net = Net()
    net(x, x)


def test_enumerate_tuple_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            index_sum = 0
            value = (x, y)
            ret = ()
            for i, j in enumerate(value):
                index_sum += i
                ret += (j,)
            return index_sum, ret

    x = Tensor(np.arange(4))
    net = Net()
    net(x, x)


def test_enumerate_tensor_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            index_sum = 0
            ret = ()
            for i, j in enumerate(x):
                index_sum += i
                ret += (j,)
            return index_sum, ret

    x = Tensor(np.arange(2 * 3).reshape(2, 3))
    net = Net()
    net(x)


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


def test_enumerate_tensor_const_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor(np.arange(2*3).reshape(2, 3))

        def construct(self):
            index_sum = 0
            ret = ()
            for i in enumerate(self.value):
                index_sum += i[0]
                ret += (i[1],)
            return index_sum, ret

    net = Net()
    net()


def test_enumerate_tuple_parameter_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            index_sum = 0
            value = (x, y)
            ret = ()
            for i in enumerate(value):
                index_sum += i[0]
                ret += (i[1],)
            return index_sum, ret

    x = Tensor(np.arange(4))
    net = Net()
    net(x, x)


def test_enumerate_tensor_parameter_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            index_sum = 0
            ret = ()
            for i in enumerate(x):
                index_sum += i[0]
                ret += (i[1],)
            return index_sum, ret

    x = Tensor(np.arange(2 * 3).reshape(2, 3))
    net = Net()
    net(x)


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


def test_enumerate_tensor_const_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor(np.arange(2 * 3).reshape(2, 3))

        def construct(self):
            index_sum = 0
            ret = ()
            for i in enumerate(self.value, 1):
                index_sum += i[0]
                ret += (i[1],)
            return index_sum, ret

    net = Net()
    net()


def test_enumerate_tuple_parameter_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            index_sum = 0
            value = (x, y)
            ret = ()
            for i in enumerate(value, 1):
                index_sum += i[0]
                ret += (i[1],)
            return index_sum, ret

    x = Tensor(np.arange(4))
    net = Net()
    net(x, x)


def test_enumerate_tensor_parameter_2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            index_sum = 0
            ret = ()
            for i, j in enumerate(x, 1):
                index_sum += i
                ret += (j,)
            return index_sum, ret

    x = Tensor(np.arange(2 * 3).reshape(2, 3))
    net = Net()
    net(x)


def test_enumerate_start_type_error():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return enumerate((x, x), start=1.2)

    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    net = Net()
    with pytest.raises(TypeError) as ex:
        net(x)
    assert "For 'enumerate', the 'start'" in str(ex.value)


def test_fallback_enumerate_with_numpy():
    """
    Feature: JIT Fallback
    Description: Test enumerate in graph mode with numpy input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.array([1, 2])
        y = enumerate(x)
        return tuple(y)

    out = foo()
    assert operator.eq(out, ((0, 1), (1, 2)))
