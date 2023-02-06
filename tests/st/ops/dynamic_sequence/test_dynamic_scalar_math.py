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
import pytest
from mindspore import context
from mindspore.nn import Cell
from tuple_help import TupleFactory

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_add():
    """
    Feature: test ScalarAdd.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x + y

    def func(x, y):
        return x + y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_sub():
    """
    Feature: test ScalarSub.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x - y

    def func(x, y):
        return x - y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()



@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_mul():
    """
    Feature: test ScalarMul.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x * y

    def func(x, y):
        return x * y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_div():
    """
    Feature: test ScalarDiv.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x / y

    def func(x, y):
        return x / y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_mod():
    """
    Feature: test ScalarMod.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x % y

    def func(x, y):
        return x % y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_floordiv():
    """
    Feature: test ScalarFloorDiv.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x // y

    def func(x, y):
        return x // y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_eq():
    """
    Feature: test ScalarEqual.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x == y

    def func(x, y):
        return x == y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_ge():
    """
    Feature: test ScalarGreaterEqual.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x >= y

    def func(x, y):
        return x >= y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_gt():
    """
    Feature: test ScalarGreater.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x > y

    def func(x, y):
        return x > y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_le():
    """
    Feature: test ScalarLessEqual.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x <= y

    def func(x, y):
        return x <= y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scalar_lt():
    """
    Feature: test ScalarLess.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x < y

    def func(x, y):
        return x < y

    net_ms = Net()
    input_x = 3
    input_y = 4
    context.set_context(mode=context.GRAPH_MODE)
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()
