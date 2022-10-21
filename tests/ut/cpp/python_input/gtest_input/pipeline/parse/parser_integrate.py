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
file: parser_integrate.py
"""
import numpy as np
import mindspore._c_expression as me
import mindspore.nn as nn
from mindspore.common import dtype
from mindspore.common.api import jit, _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.train.model import Model
from tests.ut.python.model.resnet import resnet50


def test_high_order_function(a):
    def f(g, x):
        return scalar_mul(g(x, x), g(x, x))

    return f(scalar_add, a)


def test_hof_tup(a, b):
    """Test higher order functions."""

    def f(gh, x, y):
        g, h = gh
        return scalar_mul(g(x, y), h(x, y))

    return f((scalar_add, scalar_mul), a, b)


def scalar_mul(x, y):
    """Implement `scalar_mul`."""
    return x * y


def scalar_add(x, y):
    """implement scalar_add"""
    return x + y


def test_while_2(x, y, z):
    rval = 0
    # Cannot compare to 0 or finite diff is unstable
    while x > -0.1:
        rval = rval + y
        x = x - z
    return rval


def test_nested_closure(x):
    a = x * x
    b = x + 5

    def f():
        def g():
            return a + b

        def h():
            return a * b

        return g if x < 0 else h

    return f()()


def test_functions_in_tuples(x, y):
    tup = scalar_add, scalar_mul
    f, g = tup
    return f(x, y) + g(x, y)


def test_closures_in_tuples(x, y):
    def f():
        return x * y

    def g():
        return x + y

    tup = f, g
    ff, gg = tup
    return ff() + gg()


@jit
def add(x, y):
    return x + y


def test_tensor_add():
    X = me.tensor()
    Y = me.tensor()
    X.set_dtype(dtype.float32)
    Y.set_dtype(dtype.float32)
    X = me.tensor(np.ones([2, 3]))
    Y = me.tensor(np.ones([2, 3]))
    tensor_add = add(X, Y)
    print("test tensor add")
    return tensor_add


def loss_func(x, y):
    return x - y


def optimizer(x):
    return x


def test_resetnet50_build():
    X = me.tensor()
    Y = me.tensor()
    X.set_dtype(dtype.float32)
    Y.set_dtype(dtype.float32)
    network = resnet50()
    Model(network=network, loss_fn=loss_func, optimizer=optimizer)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, bias_init='zeros')

    def construct(self, inputs):
        return self.conv(inputs)


class TestNet(nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.param = Parameter(Tensor([1, 3, 16, 50]), "param")

    def construct(self, inputs):
        self.param = self.param + inputs
        return self.param


def test_compile_conv2d():
    net = Net()
    inputs = Tensor(np.ones([1, 3, 16, 50]).astype(np.float32))
    _cell_graph_executor.compile(net, inputs)


def test_none(x, y):
    def func(x, y):
        if y is None:
            return x
        return x + y

    return func(x, y)


def test_get_attr(x):
    a = F.scalar_mul(x, x)
    return a


@jit
def known():
    return unknown()


def test_undefined_symbol():
    known()
