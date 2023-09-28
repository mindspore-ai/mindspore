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
from math import log
import pytest
import numpy as np
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops import functional as F
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE, grad_for_scalar=True)
context_prepare()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()



@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_eq():
    """
    Feature: test scalar_eq.
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_ge():
    """
    Feature: test scalar_ge.
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_gt():
    """
    Feature: test scalar_gt.
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_le():
    """
    Feature: test scalar_le.
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_lt():
    """
    Feature: test scalar_lt.
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
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bool_not():
    """
    Feature: test bool_not.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return x != y

    def func(x, y):
        return x != y

    net_ms = Net()
    input_x = 3
    input_y = 4
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_uadd():
    """
    Feature: test uadd.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x):
            return F.scalar_uadd(x)

    def func(x):
        return x

    net_ms = Net()
    input_x = 3
    fact = TupleFactory(net_ms, func, (input_x,))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_usub():
    """
    Feature: test uadd.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x):
            return F.scalar_usub(x)

    def func(x):
        return -x

    net_ms = Net()
    input_x = 3
    fact = TupleFactory(net_ms, func, (input_x,))
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_log():
    """
    Feature: test uadd.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x):
            return F.scalar_log(x)

    def func(x):
        return log(x)

    net_ms = Net()
    input_x = 8
    out = net_ms(input_x)
    expect = func(input_x)
    assert np.allclose(out, expect, rtol=1e-03, atol=1.e-8)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_pow():
    """
    Feature: test uadd.
    Description: inputs is dynamic scalar.
    Expectation: the result match with numpy result
    """
    class Net(Cell):
        def construct(self, x, y):
            return F.scalar_pow(x, y)

    def func(x, y):
        return pow(x, y)

    net_ms = Net()
    input_x = 8
    input_y = 2
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()
    fact.grad_impl()
