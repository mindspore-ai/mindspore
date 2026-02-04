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
""" test graph fallback """
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_self_attr():
    """
    Feature: JIT Fallback
    Description: Use self.attr in expressions supported by JIT Fallback.
    Expectation: No exception.
    """
    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.dim = 1

        def construct(self, x):
            batch = x.shape[0]
            one = Tensor(np.ones([batch, self.dim]), mstype.float32)
            return one * x

    net = Network()
    x = Tensor([1, 2], mstype.float32)
    out = net(x)
    expect = np.array([[1., 2.], [1., 2.]])
    assert np.allclose(out.asnumpy(), expect, 1.e-2, 1.e-2)


def test_fallback_self_attr_fn():
    """
    Feature: JIT Fallback
    Description: Use self.attr of type function in expressions supported by JIT Fallback.
    Expectation: No exception.
    """
    class Network(nn.Cell):
        def __init__(self, fn):
            super(Network, self).__init__()
            self.fn = fn

        def construct(self):
            x = np.array([1, 2, 3])
            y = np.array([3, 4, 5])
            out = Tensor(self.fn(x, y))
            return out

    def fn(x, y):
        return x + y

    net = Network(fn)
    out = net()
    expect = np.array([4, 6, 8])
    assert np.all(out.asnumpy() == expect)


def test_fallback_self_attr_attr():
    """
    Feature: JIT Fallback
    Description: In expressions supported by JIT Fallback, use the attribute of self.attr.
    Expectation: No exception.
    """
    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.value = [2, 2, 3]

        def construct(self):
            x = np.array(self.value.count(2))
            return Tensor(x)

    net = Network()
    out = net()
    assert out == 2


def test_fallback_self_method():
    """
    Feature: JIT Fallback
    Description: Use self.method in expressions supported by JIT Fallback.
    Expectation: No exception.
    """
    class Network(nn.Cell):
        def construct(self):
            x = np.array([1, 2, 3])
            y = np.array([3, 4, 5])
            out = Tensor(self.fn(x, y))
            return out

        def fn(self, x, y):
            return x + y

    net = Network()
    out = net()
    expect = np.array([4, 6, 8])
    assert np.all(out.asnumpy() == expect)


def test_fallback_self_method_2():
    """
    Feature: JIT Fallback
    Description: Add with numpy array should be converted to PyInterpret node first.
    Expectation: No exception.
    """
    class Network(nn.Cell):
        def construct(self):
            x = np.array([1, 2, 3])
            y = np.array([3, 4, 5])
            z = self.fn(x, y)
            out = Tensor(z)
            return out

        def fn(self, x, y):
            return x + y

    net = Network()
    out = net()
    expect = np.array([4, 6, 8])
    assert np.all(out.asnumpy() == expect)


def test_fallback_self_variable():
    """
    Feature: JIT Fallback
    Description: Use self as variable name
    Expectation: No exception
    """
    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.value = 5

        def construct(self):
            x = self.value
            self = 10 # pylint: disable=W0642
            return x, Tensor(self)

    net = Network()
    out_x, out_self = net()
    assert out_x == 5
    assert out_self == 10


def test_fallback_self_variable_with_get_attr():
    """
    Feature: JIT Fallback
    Description: Use self as variable name
    Expectation: No exception
    """
    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.value = 5

        def construct(self):
            x = getattr(self, 'value')
            self = 10 # pylint: disable=W0642
            return x, Tensor(self)

    net = Network()
    out_x, out_self = net()
    assert out_x == 5
    assert out_self == 10
