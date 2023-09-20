# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test jit forbidden api in graph mode. """
import pytest

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import context, jit, Tensor
from mindspore.common.initializer import initializer, One
from mindspore.common.parameter import Parameter
from mindspore import log as logger

context.set_context(mode=context.GRAPH_MODE)


def test_jit_forbidden_api_one1():
    """
    Feature: mindspore.common.initializer.One
    Description: test jit forbidden api 'One' in graph mode.
    Expectation: throw RuntimeError
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            t = One()
            return t

    net = Net()
    with pytest.raises(RuntimeError) as ex:
        net()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the class 'mindspore.common.initializer.One'" in str(ex.value)


def test_jit_forbidden_api_one2():
    """
    Feature: mindspore.common.initializer.One
    Description: test jit forbidden api 'One' in graph mode.
    Expectation: throw RuntimeError
    """
    @jit
    def foo():
        t = One()
        return t

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the class 'mindspore.common.initializer.One'" in str(ex.value)


def test_jit_forbidden_api_initializer1():
    """
    Feature: mindspore.common.initializer.initializer
    Description: test jit forbidden api 'initializer' in graph mode.
    Expectation: throw RuntimeError
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            t = initializer('ones', [1, 2, 3], mstype.float32)
            return t

    net = Net()
    with pytest.raises(RuntimeError) as ex:
        net()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the method or function 'mindspore.common.initializer.initializer'" in str(ex.value)


def test_jit_forbidden_api_initializer2():
    """
    Feature: mindspore.common.initializer.initializer
    Description: test jit forbidden api 'initializer' in graph mode.
    Expectation: throw RuntimeError
    """
    @jit
    def foo():
        t = initializer('ones', [1, 2, 3], mstype.float32)
        return t

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the method or function 'mindspore.common.initializer.initializer'" in str(ex.value)


def test_jit_forbidden_api_untrainable_params1():
    """
    Feature: mindspore.nn.cell.Cell.untrainable_params
    Description: test jit forbidden api 'untrainable_params' in graph mode.
    Expectation: throw RuntimeError
    """
    class InnerNet(nn.Cell):
        def construct(self):
            return True

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.untrainable_params()
            return out

    net = Net()
    with pytest.raises(RuntimeError) as ex:
        net()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the method or function 'mindspore.nn.cell.Cell.untrainable_params'" in str(ex.value)


def test_jit_forbidden_api_untrainable_params2():
    """
    Feature: mindspore.nn.cell.Cell.untrainable_params
    Description: test jit forbidden api 'untrainable_params' in graph mode.
    Expectation: throw RuntimeError
    """
    class Net(nn.Cell):
        def construct(self):
            return True

    @jit
    def foo():
        return Net().untrainable_params()

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the method or function 'mindspore.nn.cell.Cell.untrainable_params'" in str(ex.value)


def test_jit_forbidden_api_get_parameters1():
    """
    Feature: mindspore.nn.cell.Cell.get_parameters
    Description: test jit forbidden api 'get_parameters' in graph mode.
    Expectation: throw RuntimeError
    """
    class InnerNet(nn.Cell):
        def construct(self):
            return True

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.get_parameters()
            return out

    net = Net()
    with pytest.raises(RuntimeError) as ex:
        net()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the method or function 'mindspore.nn.cell.Cell.get_parameters'" in str(ex.value)


def test_jit_forbidden_api_get_parameters2():
    """
    Feature: mindspore.nn.cell.Cell.untrainable_params
    Description: test jit forbidden api 'get_parameters' in graph mode.
    Expectation: throw RuntimeError
    """
    class Net(nn.Cell):
        def construct(self):
            return True

    @jit
    def foo():
        return Net().get_parameters()

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the method or function 'mindspore.nn.cell.Cell.get_parameters'" in str(ex.value)


def test_jit_forbidden_api_type():
    """
    Feature: Check JIT Forbidden API
    Description: Test api does not has attribute '__module__' in graph mode.
    Expectation: No Expectation
    """
    @jit
    def foo():
        x = type("C", (object,), {})
        return x

    foo()


def test_jit_forbidden_method_tensor_set_const_arg():
    """
    Feature: Check JIT Forbidden API
    Description: Test 'Tensor' does not support the method 'set_const_arg' in graph mode.
    Expectation: No Expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            x.set_const_arg(False)
            return x

    x = Tensor([1, 2, 3], dtype=mstype.float32, const_arg=True)
    net = Net()
    with pytest.raises(RuntimeError) as ex:
        net(x)
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the 'Tensor' object's method 'set_const_arg' is not supported" in str(ex.value)


def test_jit_forbidden_method_parameter_set_data():
    """
    Feature: Check JIT Forbidden API
    Description: Test 'Parameter' does not support the method 'set_data' in graph mode.
    Expectation: No Expectation
    """
    class Net(nn.Cell):
        def __init__(self, initial_input):
            super(Net, self).__init__()
            self.initial_input = initial_input
            self.weight = Parameter(initial_input, name="abc")

        def construct(self, construct_input):
            self.weight.set_data(construct_input)
            return construct_input

    x = Tensor([1, 2, 3], dtype=mstype.float32)
    y = Tensor([4, 5, 6], dtype=mstype.float32)
    net = Net(x)
    with pytest.raises(RuntimeError) as ex:
        net(y)
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)
    assert "the 'Parameter' object's method 'set_data' is not supported" in str(ex.value)


def test_jit_forbidden_api_tensor_assign_value():
    """
    Feature: mindspore.common.tensor
    Description: test jit forbidden api 'assign_value' in graph mode.
    Expectation: throw RuntimeError
    """
    class Net(nn.Cell):
        def construct(self, input_x, input_y):
            return input_x.assign_value(input_y)

    net = Net()
    with pytest.raises(RuntimeError) as ex:
        x = Tensor([1, 2, 3, 4])
        y = Tensor(np.array([[1, 2], [3, 4]]))
        output = net(x, y)
        assert x == output
    assert "Failed to compile in GRAPH_MODE because the 'Tensor' object's method 'assign_value'" in str(ex.value)


def test_jit_forbidden_api_context_get_context():
    """
    Feature: mindspore.context
    Description: test jit forbidden api 'context.get_context' in graph mode.
    Expectation: throw RuntimeError
    """
    context.set_context(mode=context.GRAPH_MODE)
    class ContextNet(nn.Cell):
        def __init__(self):
            super(ContextNet, self).__init__()

        def construct(self, x):
            if not context.get_context('mode'):
                x = x - 1
            else:
                x = x + x
            return x
    net = ContextNet()
    inputs = Tensor(np.random.randn(6, 3, 6, 6).astype(np.float32))
    with pytest.raises(RuntimeError) as ex:
        net(inputs)
    assert "Failed to compile in GRAPH_MODE because the method "\
        "or function 'mindspore.context.get_context'" in str(ex.value)


def test_jit_forbidden_api_log_warning():
    """
    Feature: mindspore.log
    Description: test jit forbidden api 'log.warning' in graph mode.
    Expectation: throw RuntimeError
    """
    context.set_context(mode=context.GRAPH_MODE)
    class LogWarningNet(nn.Cell):
        def __init__(self):
            super(LogWarningNet, self).__init__()

        def construct(self, x):
            logger.warning("some warnings.")
            return x
    net = LogWarningNet()
    inputs = Tensor(np.random.randn(6, 3, 6, 6).astype(np.float32))
    with pytest.raises(RuntimeError) as ex:
        net(inputs)
    assert "Failed to compile in GRAPH_MODE because the method "\
        "or function 'mindspore.log.warning'" in str(ex.value)
