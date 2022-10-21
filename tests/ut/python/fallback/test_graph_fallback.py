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
""" test graph fallback """
import functools
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, jit, context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.probability import distribution
import mindspore.common.dtype as mstype
import mindspore.common._monad as monad
import mindspore.scipy.linalg as alg

context.set_context(mode=context.GRAPH_MODE)

# `add_func` is defined in current file.
def add_func(x, y):
    return x + y


@jit
def do_increment(i):
    add_1 = F.partial(add_func, 1)
    return add_1(i)


def test_increment():
    a = do_increment(9)
    assert a == 10


@jit
def use_monad(x, y):
    res = P.Mul()(x, y)
    res = F.depend(res, monad.U)
    return res


def test_use_monad():
    x = Tensor(1.0, mstype.float32)
    y = Tensor(1.0, mstype.float32)
    print(use_monad(x, y))


@jit
def use_tuple_of_tensor():
    me_x = (Tensor(1), Tensor(1))
    return me_x


def test_tuple_of_tensor():
    """
    Feature: JIT Fallback
    Description: Test tuple of tensor in graph mode.
    Expectation: No exception.
    """
    print(use_tuple_of_tensor())


@jit
def use_list_of_tensor():
    me_x = [Tensor(1), Tensor(1)]
    return me_x


def test_list_of_tensor():
    """
    Feature: JIT Fallback
    Description: Test list of tensor in graph mode.
    Expectation: No exception.
    """
    print(use_list_of_tensor())


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.x = Tensor([2, 3, 4])

    def construct(self):
        x_len = len(self.x)
        for i in range(x_len):
            print(i)
        return x_len


def test_builtins_len():
    net = Net()
    net()


@jit
def np_fallback_func():
    array_x = tuple([2, 3, 4, 5])
    np_x = np.array(array_x).astype(np.float32)
    me_x = Tensor(np_x)
    me_x = me_x + me_x
    return me_x


def test_np_fallback_func():
    print(np_fallback_func())


# Test `return` interpret node.
@jit
def div_mod_func1():
    x = 8
    y = 3
    a = divmod(x, y)
    return Tensor(a)


def test_div_mod_func1():
    print(div_mod_func1())  # (2, 2)


# Test interpret node with parameters as input.
@jit
def div_mod_func2(x, y):
    a = divmod(x, y)
    return Tensor(a)


def test_div_mod_func2_scalar():
    """
    Feature: JIT Fallback
    Description: Test divmod in graph.
    Expectation: No exception.
    """
    print(div_mod_func2(8, 3))  # (2, 2)


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_div_mod_func2_tensor():
    """
    Feature: JIT Fallback
    Description: Test divmod with Tensor input in graph. We'll support it in Tensor Input Fallback solution.
    Expectation: Not supported exception.
    """
    with pytest.raises(RuntimeError) as err:
        print(div_mod_func2(Tensor(8), Tensor(3)))
    assert "Not support Tensor or variable type as input during running JIT Fallback, but got" in str(err.value)


@jit
def select_func(cond, x, y):
    if isinstance(cond, (tuple, list)):
        output = y
    elif isinstance(cond, Tensor):
        output = F.select(cond, x, y)
    else:
        output = x
    return output


def test_select_func():
    cond = Tensor([True, False])
    x = Tensor([2, 3], mstype.float32)
    y = Tensor([1, 2], mstype.float32)
    print(select_func(cond, x, y))


@jit
def select_func2(cond, x, y):
    if isinstance(cond, (tuple, list)):
        output = y
    if isinstance(cond, Tensor):
        output = F.select(cond, x, y)
    else:
        output = x
    return output


def test_select_func2():
    cond = Tensor([True, False])
    x = Tensor([2, 3], mstype.float32)
    y = Tensor([1, 2], mstype.float32)
    print(select_func2(cond, x, y))


@jit
def slice_func(a, b):
    a[1:3, ::] = b
    return a


def test_slice_func():
    a = Tensor(np.arange(60).reshape(3, 4, 5), dtype=mstype.float32)
    b = Tensor([1], dtype=mstype.float32)
    print(slice_func(a, b))


def test_context():
    """
    Feature: JIT Fallback
    Description: Test context in graph.
    Expectation: No exception.
    """
    class ContextNet(nn.Cell):
        def __init__(self):
            super(ContextNet, self).__init__()
            self.mode = context.get_context("mode")

        def construct(self):
            out = 1
            if self.mode == context.GRAPH_MODE:
                out = 2
            return out

    net = ContextNet()
    out = net()
    print(out)


def test_scipy_module():
    """
    Feature: JIT Fallback
    Description: Test scipy module in graph.
    Expectation: No exception.
    """
    class Network(nn.Cell):
        def construct(self, x):
            return alg.eigh(x)

    net = Network()
    x = Tensor([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
    out = net(x)
    print(out)


def test_probability_cauchy():
    """
    Feature: JIT Fallback
    Description: NumPy method is called in probability cauchy.
    Expectation: No exception.
    """
    class CauchyProb(nn.Cell):
        def __init__(self, loc, scale, seed=10, dtype=mstype.float32, name='Cauchy'):
            super().__init__()
            self.b = distribution.Cauchy(loc, scale, seed, dtype, name)

        def construct(self, value, loc=None, scale=None):
            out1 = self.b.prob(value, loc, scale)
            out2 = self.b.log_prob(value, loc, scale)
            out3 = self.b.cdf(value, loc, scale)
            out4 = self.b.log_cdf(value, loc, scale)
            out5 = self.b.survival_function(value, loc, scale)
            out6 = self.b.log_survival(value, loc, scale)
            return out1, out2, out3, out4, out5, out6


    loc = np.random.randn(1024, 512, 7, 7).astype(np.float32)
    scale = np.random.uniform(0.0001, 100, size=(1024, 512, 7, 7)).astype(np.float32)
    loc_a = np.random.randn(1024, 512, 7, 7).astype(np.float32)
    scale_a = np.random.uniform(0.0001, 100, size=(1024, 512, 7, 7)).astype(np.float32)
    value = np.random.randn(1024, 512, 7, 7).astype(np.float32)

    net = CauchyProb(loc, scale)
    net(Tensor(value), Tensor(loc_a), Tensor(scale_a))


def test_third_party_module_functools():
    """
    Feature: JIT Fallback
    Description: functools is a python built-in module and does not perform JIT Fallback.
    Expectation: No exception.
    """
    class ModuleNet(nn.Cell):
        def construct(self, x, y):
            func = functools.partial(add_func, x)
            out = func(y)
            return out

    x = Tensor([1, 2, 3], mstype.int32)
    y = Tensor([4, 5, 6], mstype.int32)
    net = ModuleNet()
    out = net(x, y)
    print(out)
