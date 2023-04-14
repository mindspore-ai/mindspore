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
""" test jit_class """
import pytest
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, jit_class

context.set_context(mode=context.GRAPH_MODE)


def test_ms_class_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of user-defined classes decorated with jit_class.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        def __init__(self):
            self.number = Tensor(1, dtype=mstype.int32)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.number
            return out

    net = Net()
    out = net()
    assert out.asnumpy() == 1


def test_ms_class_input_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of user-defined classes decorated with jit_class.
    Expectation: No exception.
    """
    @jit_class
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


def test_ms_class_input_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of user-defined classes decorated with jit_class.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        def __init__(self):
            self.val = Tensor(2, dtype=mstype.int32)

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
    assert out.asnumpy() == 6


def test_ms_class_nested():
    """
    Feature: JIT Fallback
    Description: Test nested jit_class in graph.
    Expectation: No exception.
    """
    @jit_class
    class Inner:
        def __init__(self):
            self.number = Tensor(1, dtype=mstype.int32)

    @jit_class
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
    assert out.asnumpy() == 1


def test_ms_class_cell_nested():
    """
    Feature: JIT Fallback
    Description: Test nested jit_class and cell in graph.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def construct(self, x):
            return x + self.val

    @jit_class
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


def test_ms_class_type_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of class type.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        val = Tensor(2, dtype=mstype.int32)

        def act(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet

        # Support accessing attributes of class type, but do not support
        # accessing methods, e.g. self.inner_net.act(1, 2)
        def construct(self):
            out = self.inner_net.val
            return out

    net = Net()
    out = net()
    assert out == 2


def test_ms_class_create_instance_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of the created class instance.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        def __init__(self, val):
            self.number = val + 3

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet

        def construct(self, x):
            net = self.inner_net(x)
            return net.number

    net = Net()
    out = net(2)
    assert out == 5


def test_raise_error_not_class_type():
    """
    Feature: JIT Fallback
    Description: Decorator jit_class cannot be used for non-class types.
    Expectation: No exception.
    """
    with pytest.raises(TypeError):
        @jit_class
        def func(x, y):
            return x + y

        func(1, 2)


def test_raise_error_decorate_cell():
    """
    Feature: JIT Fallback
    Description: Decorator jit_class cannot be used for nn.Cell
    Expectation: No exception.
    """
    with pytest.raises(TypeError):
        @jit_class
        class Net(nn.Cell):
            def construct(self, x):
                return x

        x = Tensor(1)
        net = Net()
        net(x)


def test_with_as_exception():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    @jit_class
    class Sample():
        def __init__(self):
            super(Sample, self).__init__()
            self.num = Tensor([1])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            print("type:", exc_type)
            print("value:", exc_value)
            print("trace:", traceback)
            return self.do_something(1)

        def do_something(self, x):
            bar = 2 / 0 + x + self.num
            return bar + 10

    class TestNet(nn.Cell):
        def construct(self, x):
            a = 1
            with Sample() as sample:
                a = sample.do_something(a + x)
            return x * a

    with pytest.raises(ValueError) as as_exception:
        x = Tensor([1])
        test_net = TestNet()
        res = test_net(x)
        print("res:", res)
        assert res == 10
    assert "The divisor could not be zero" in str(as_exception.value)
