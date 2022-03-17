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
import pytest
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, ms_class, ms_function
from . import test_graph_fallback

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_self_attr():
    """
    Feature: JIT Fallback
    Description: Test self.attr in graph.
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
    Description: Test self.attr in graph.
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
    Description: Test self.attr in graph.
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
    Description: Test self.method in graph.
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


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_fallback_self_method_tensor():
    """
    Feature: JIT Fallback
    Description: Test self.method in graph.
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
    print(out)


def test_fallback_import_modules():
    """
    Feature: JIT Fallback
    Description: add_func is defined in test_graph_fallback.py
    Expectation: No exception.
    """
    @ms_function
    def use_imported_module(x, y):
        out = test_graph_fallback.add_func(x, y)
        return out

    x = Tensor(2, dtype=mstype.int32)
    y = Tensor(3, dtype=mstype.int32)
    out = use_imported_module(x, y)
    print(out)


def test_fallback_class_attr():
    """
    Feature: JIT Fallback
    Description: Test user-defined class attributes in graph.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.number = 1

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.number
            return out

    net = Net()
    out = net()
    assert out == 1


def test_fallback_class_method():
    """
    Feature: JIT Fallback
    Description: Test user-defined class methods in graph.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.val = 2

        def act(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.act(1, 2)
            return out

    net = Net()
    out = net()
    assert out == 6


def test_fallback_class_input_attr():
    """
    Feature: JIT Fallback
    Description: Test user-defined class attributes in graph.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.number = Tensor(np.array([1, 2, 3]))

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
            self.inner_net = net()

        def construct(self):
            out = self.inner_net.number
            return out

    net = Net(InnerNet)
    out = net()
    expect_res = np.array([1, 2, 3])
    assert np.all(out.asnumpy() == expect_res)


def test_fallback_class_input_method():
    """
    Feature: JIT Fallback
    Description: Test user-defined class methods in graph.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.val = 2

        def act(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
            self.inner_net = net()

        def construct(self):
            out = self.inner_net.act(1, 2)
            return out

    net = Net(InnerNet)
    out = net()
    assert out == 6


def test_fallback_class_class_nested():
    """
    Feature: JIT Fallback
    Description: Test nested ms_class in graph.
    Expectation: No exception.
    """
    @ms_class
    class Inner:
        def __init__(self):
            self.number = 1

    @ms_class
    class InnerNet:
        def __init__(self):
            self.inner = Inner()

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.inner.number
            return out

    net = Net()
    out = net()
    assert out == 1


def test_fallback_class_cell_nested():
    """
    Feature: JIT Fallback
    Description: Test nested ms_class and cell in graph.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def construct(self, x):
            return x + self.val

    @ms_class
    class TrainNet():
        class Loss(nn.Cell):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def construct(self, x):
                out = self.net(x)
                return out * 2

        def __init__(self, net):
            self.net = net
            loss_net = self.Loss(self.net)
            self.number = loss_net(10)

    global_net = Net(1)
    class LearnNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.value = TrainNet(global_net).number

        def construct(self, x):
            return x + self.value

    leanrn_net = LearnNet()
    out = leanrn_net(3)
    print(out)
    assert out == 25


@pytest.mark.skip(reason='Not support in graph yet')
def test_fallback_class_isinstance():
    """
    Feature: JIT Fallback
    Description: Test ms_class in graph.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.number = 1

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self, x):
            if isinstance(self.inner_net, InnerNet):
                return x + 10
            return x

    net = Net()
    out = net(5)
    assert out == 15


def test_fallback_raise_error_not_class_type():
    """
    Feature: JIT Fallback
    Description: Test ms_class in graph.
    Expectation: No exception.
    """
    with pytest.raises(TypeError):
        @ms_class
        def func(x, y):
            return x + y

        func(1, 2)


def test_fallback_raise_error_not_class_instance():
    """
    Feature: JIT Fallback
    Description: Test ms_class in graph.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.number = 1

    class Net(nn.Cell):
        def construct(self):
            out = InnerNet().number
            return out

    with pytest.raises(ValueError):
        net = Net()
        net()


def test_fallback_raise_error_decorate_cell():
    """
    Feature: JIT Fallback
    Description: Test ms_class in graph.
    Expectation: No exception.
    """
    @ms_class
    class Net(nn.Cell):
        def construct(self, x):
            return x

    with pytest.raises(TypeError):
        x = Tensor(1)
        net = Net()
        net(x)
