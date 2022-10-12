# Copyright 2022 Huawei Technologies Co., Ltd
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
import ge_train_env  # pylint: disable=unused-import
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def test_convert_return():
    """
    Feature: convert ge graph
    Description: test Return node
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()

        def construct(self, x_, y_):
            return self.add(x_, y_)

    x = np.ones([1, 3, 3, 4]).astype(np.float64)
    y = np.ones([1, 3, 3, 4]).astype(np.float64)
    add = Net()
    output = add(Tensor(x), Tensor(y))
    expect = np.add(x, y)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-5, atol=1e-5)


def test_convert_update_state():
    """
    Feature: convert ge graph
    Description: test UpdateState node
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self, para):
            super(Net, self).__init__()
            self.para = Parameter(para, name="para")
            self.assign_add = P.AssignAdd()

        def construct(self, value):
            self.assign_add(self.para, value)
            return self.para

    x = Tensor(1, dtype=mstype.int32)
    y = Tensor(2, dtype=mstype.int32)
    expect = Tensor(3, dtype=mstype.int32)
    net = Net(x)
    out = net(y)
    assert out == expect


def test_convert_load():
    """
    Feature: convert ge graph
    Description: test Load node
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign = P.Assign()
            self.variable = Parameter(Tensor(0, ms.float32), name="global")

        def construct(self, x):
            out = self.variable + x
            self.assign(self.variable, 1)
            out = self.variable + out
            return out

    x = Tensor([2], ms.float32)
    net = Net()
    out = net(x)
    assert out == 3


def test_convert_make_tuple():
    """
    Feature: convert ge graph
    Description: test MakeTuple node
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.AddN()

        def construct(self, x, y):
            return self.add((x, y))

    x = np.random.randn(1, 3, 3, 4).astype(np.float64)
    y = np.random.randn(1, 3, 3, 4).astype(np.float64)
    add = Net()
    output = add(Tensor(x), Tensor(y))
    expect = np.add(x, y)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-5, atol=1e-5)


def test_convert_tuple_get_item():
    """
    Feature: convert ge graph
    Description: test TupleGetItem node
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sort = P.Sort(axis=1)

        def construct(self, x):
            y = self.sort(x)
            return y[0]

    x = np.random.random((3, 3)).astype(np.float32)
    net = Net()
    output = net(Tensor(x, ms.float32))
    expect = np.sort(x, axis=1)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


def test_convert_make_tuple_make_tuple():
    """
    Feature: convert ge graph
    Description: test MakeTuple's input is MakeTuple
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()

        def construct(self, x_, y_):
            result1 = []
            result2 = []
            x1 = self.add(x_, y_)
            x2 = self.add(x1, y_)
            result1.append(x1)
            result1.append(x2)
            result2.append(x1)
            result2.append(result1)
            return result2

    x = np.ones([1]).astype(np.int32)
    y = np.ones([1]).astype(np.int32)
    add = Net()
    output = add(Tensor(x), Tensor(y))

    assert output[0].asnumpy() == 2
    assert output[1][0].asnumpy() == 2
    assert output[1][1].asnumpy() == 3
