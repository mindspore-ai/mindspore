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
test_structure_output
"""
import pytest
import numpy as np

import mindspore.ops.operations as P
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops.functional import depend

context.set_context(mode=context.GRAPH_MODE)


def test_output_const_tuple_0():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = (1, 2, 3)

        def construct(self):
            return self.x

    x = (1, 2, 3)
    net = Net()
    assert net() == x


def test_output_const_tuple_1():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.tuple_1 = (1, 2, 3)
            self.tuple_2 = (4, 5, 6)

        def construct(self):
            ret = self.tuple_1 + self.tuple_2
            return ret

    net = Net()
    assert net() == (1, 2, 3, 4, 5, 6)


def test_output_const_list():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.tuple_1 = [1, 2, 3]

        def construct(self):
            ret = self.tuple_1
            return ret

    net = Net()
    assert net() == [1, 2, 3]


def test_output_const_int():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.number_1 = 2
            self.number_2 = 3

        def construct(self):
            ret = self.number_1 + self.number_2
            return ret

    net = Net()
    assert net() == 5


def test_output_const_str():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.str = "hello world"

        def construct(self):
            ret = self.str
            return ret

    net = Net()
    assert net() == "hello world"


def test_output_parameter_int():
    class Net(Cell):
        def construct(self, x):
            return x

    x = Tensor(np.array(88).astype(np.int32))
    net = Net()
    assert net(x) == x


def test_output_parameter_str():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = "hello world"

        def construct(self):
            return self.x

    x = "hello world"
    net = Net()
    assert net() == x


def test_output_function():
    """
    Feature: Graph syntax.
    Description: Return function as output in graph mode.
    Expectation: Throw Exception.
    """
    class Net(Cell):
        def func(self):
            return 0

        def construct(self):
            return self.func

    with pytest.raises(RuntimeError, match="Function in output is not supported."):
        net = Net()
        net()


def test_output_function_tuple():
    """
    Feature: Graph syntax.
    Description: Return function in output in graph mode.
    Expectation: Throw Exception.
    """
    class Net(Cell):
        def func(self):
            return 0

        def construct(self):
            return (self.func,)

    with pytest.raises(RuntimeError, match="Function in output is not supported."):
        net = Net()
        net()


def test_output_getattr_function_tuple():
    """
    Feature: Graph syntax.
    Description: Return function in output in graph mode.
    Expectation: Throw Exception.
    """
    class Net(Cell):
        def construct(self, x):
            abs_x = getattr(x, 'abs')
            abs_y = getattr(Tensor([-1, -2, -3]), 'abs')
            return abs_x, abs_y

    with pytest.raises(RuntimeError, match="Function in output is not supported."):
        x = Tensor([-1, -2, -3])
        net = Net()
        net(x)


def test_tuple_tuple_0():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()
            self.sub = P.Sub()

        def construct(self, x, y):
            xx = self.add(x, x)
            yy = self.add(y, y)
            xxx = self.sub(x, x)
            yyy = self.sub(y, y)
            ret = ((xx, yy), (xxx, yyy))
            ret = (ret, ret)
            return ret

    net = Net()
    x = Tensor(np.ones([2], np.int32))
    y = Tensor(np.zeros([3], np.int32))
    net(x, y)


def test_tuple_tuple_1():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()
            self.sub = P.Sub()

        def construct(self, x, y):
            xx = self.add(x, x)
            yy = self.add(y, y)
            ret = ((xx, yy), x)
            ret = (ret, ret)
            return ret

    net = Net()
    x = Tensor(np.ones([2], np.int32))
    y = Tensor(np.zeros([3], np.int32))
    net(x, y)


def test_tuple_tuple_2():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.relu = P.ReLU()
            self.depend = depend

        def construct(self, x, y):
            xx = self.add(x, x)
            yy = self.add(y, y)
            xxx = self.sub(x, x)
            yyy = self.sub(y, y)
            z = self.relu(x)
            ret = ((xx, yy), (xxx, yyy))
            ret = (ret, ret)
            ret = self.depend(ret, z)
            return ret

    net = Net()
    x = Tensor(np.ones([2], np.int32))
    y = Tensor(np.zeros([3], np.int32))
    net(x, y)


def test_tuple_tuple_3():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.relu = P.ReLU()
            self.depend = depend

        def construct(self, x, y):
            xx = self.add(x, x)
            yy = self.add(y, y)
            z = self.relu(x)
            ret = ((xx, yy), x)
            ret = (ret, ret)
            ret = self.depend(ret, z)
            return ret

    net = Net()
    x = Tensor(np.ones([2], np.int32))
    y = Tensor(np.zeros([3], np.int32))
    net(x, y)


def test_soft():
    class SoftmaxCrossEntropyWithLogitsNet(Cell):
        def __init__(self):
            super(SoftmaxCrossEntropyWithLogitsNet, self).__init__()
            self.soft = P.SoftmaxCrossEntropyWithLogits()
            self.value = (Tensor(np.zeros((2, 2)).astype(np.float32)), Tensor(np.ones((2, 2)).astype(np.float32)))

        def construct(self, x, y, z):
            xx = x + y
            yy = x - y
            ret = self.soft(xx, yy)
            ret = (ret, z)
            ret = (ret, self.value)
            return ret

    input1 = Tensor(np.zeros((2, 2)).astype(np.float32))
    input2 = Tensor(np.ones((2, 2)).astype(np.float32))
    input3 = Tensor((np.ones((2, 2)) + np.ones((2, 2))).astype(np.float32))
    net = SoftmaxCrossEntropyWithLogitsNet()
    net(input1, input2, input3)


def test_const_depend():
    class ConstDepend(Cell):
        def __init__(self):
            super(ConstDepend, self).__init__()
            self.value = (Tensor(np.zeros((2, 3)).astype(np.float32)), Tensor(np.ones((2, 3)).astype(np.float32)))
            self.soft = P.SoftmaxCrossEntropyWithLogits()
            self.depend = depend

        def construct(self, x, y, z):
            ret = x + y
            ret = ret * z
            ret = self.depend(self.value, ret)
            ret = (ret, self.soft(x, y))
            return ret

    input1 = Tensor(np.zeros((2, 2)).astype(np.float32))
    input2 = Tensor(np.ones((2, 2)).astype(np.float32))
    input3 = Tensor((np.ones((2, 2)) + np.ones((2, 2))).astype(np.float32))
    net = ConstDepend()
    net(input1, input2, input3)
