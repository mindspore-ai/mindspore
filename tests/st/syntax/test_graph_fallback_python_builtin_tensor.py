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
"""test python built-in functions in graph mode"""
import operator
import pytest
import numpy as np
from mindspore import Tensor, context, nn, ms_function
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_abs_tensor():
    """
    Feature: JIT Fallback
    Description: Test abs(Tensor) with a variable tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self, y):
            x = Tensor([-1, 2])
            return abs(x + y)

    net = Net()
    assert np.all(net(Tensor([-1, 2])).asnumpy() == np.array([2, 4]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_all_tensor():
    """
    Feature: JIT Fallback
    Description: Test all(Tensor) with a variable tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return all(x), all(y)

    net = Net()
    x = Tensor(np.array([0, 1, 2, 3]))
    y = Tensor(np.array([1, 1]))
    out1, out2 = net(x, y)
    assert (not out1) and out2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_all_tensor_constant():
    """
    Feature: JIT Fallback
    Description: Test all(Tensor) with a constant tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self):
            x = Tensor(np.array([0, 1, 2, 3]))
            y = Tensor(np.array([1, 1]))
            return all(x), all(y)

    net = Net()
    out1, out2 = net()
    assert (not out1) and out2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_any_tensor():
    """
    Feature: JIT Fallback
    Description: Test any(Tensor) with a variable tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return any(x), any(y)

    net = Net()
    x = Tensor(np.array([0, 0]))
    y = Tensor(np.array([1, 0]))
    out1, out2 = net(x, y)
    assert (not out1) and out2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_any_tensor_constant():
    """
    Feature: JIT Fallback
    Description: Test any(Tensor) with a constant tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self):
            x = Tensor(np.array([0, 0]))
            y = Tensor(np.array([1, 0]))
            return any(x), any(y)

    net = Net()
    out1, out2 = net()
    assert (not out1) and out2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_round_tensor():
    """
    Feature: JIT Fallback
    Description: Test round(Tensor) with a variable tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self, x):
            return round(x)

    net = Net()
    x = Tensor(np.array([0.1, 4.51, 9.9]), mstype.float32)
    out = net(x)
    expect = Tensor(np.array([0.0, 5.0, 10.0]))
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_round_tensor_constant():
    """
    Feature: JIT Fallback
    Description: Test any(Tensor) with a constant tensor in graph mode
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            x = Tensor(np.array([0.1, 4.51, 9.9]), mstype.float32)
            return round(x)

    net = Net()
    out = net()
    expect = Tensor(np.array([0.0, 5.0, 10.0]))
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_builtin_function_max_min_with_tensor():
    """
    Feature: Support the type of the input of built-in function max is tensor.
    Description: Support the type of the input of built-in function max is tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo(x, y):
        return max(x, y), min(x, y)

    max_out, min_out = foo(Tensor([1]), Tensor([2]))
    assert operator.eq(max_out, 2)
    assert operator.eq(min_out, 1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_builtin_function_max_min_with_multiple_tensor():
    """
    Feature: Support the type of the input of built-in function max is tensor.
    Description: Support the type of the input of built-in function max is tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo(x, y, z):
        return max(x, y, z), min(x, y, z)

    max_out, min_out = foo(Tensor([1]), Tensor([2]), Tensor([3]))
    assert operator.eq(max_out, 3)
    assert operator.eq(min_out, 1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_builtin_function_max_min_with_tensor_list():
    """
    Feature: Support the type of the input of built-in function min is tensor.
    Description: Support the type of the input of built-in function min is tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo(x):
        return min(x), max(x)

    min_out, max_out = foo(Tensor([1, 2, 3, 4, 5], dtype=mstype.float32))
    assert operator.eq(min_out, 1)
    assert operator.eq(max_out, 5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_tensor_n_default_1():
    """
    Feature: JIT Fallback
    Description: Description: Test sum(Tensor) in graph mode with tensor and input n is default.
    Expectation: No exception
    """
    class Net(nn.Cell):
        def construct(self, x):
            return sum(x)

    net = Net()
    x = Tensor([3, 4, 5], dtype=mstype.float32)
    out = net(x)
    assert out == 12


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_tensor_n_default_2():
    """
    Feature: JIT Fallback
    Description: Description: Test sum(Tensor) in graph mode with tensor and input n is default.
    Expectation: No exception
    """
    class Net(nn.Cell):
        def construct(self, x):
            return sum(x)

    net = Net()
    x = Tensor([[1, 2], [3, 4]], dtype=mstype.float32)
    out = net(x)
    assert np.allclose(out.asnumpy(), np.array([4, 6]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_with_x_tensor_n_not_default_1():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tensor and input n not default.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return sum(x, y)

    net = Net()
    x, y = Tensor([3, 4, 5], dtype=mstype.float32), 4
    out = net(x, y)
    assert out == 16


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_with_x_tensor_n_not_default_2():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tensor and input n not default.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return sum(x, y)

    net = Net()
    x, y = Tensor([[1, 2], [3, 4]], dtype=mstype.float32), [5, 6]
    out = net(x, y)
    assert np.allclose(out.asnumpy(), np.array([9, 12]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_with_x_list_of_tensor():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode when input x is list of tensor.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return sum(x, y)

    net = Net()
    x, y = [1, Tensor([[1, 2], [3, 4]]), Tensor([[1, 2], [3, 4]])], Tensor([[1, 1], [1, 1]])
    out = net(x, y)
    assert np.allclose(out.asnumpy(), np.array([[4, 6], [8, 10]]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_list_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list(Tensor([1, 2, 3]))
        x.append(Tensor([4]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 4
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3
    assert isinstance(out[3], Tensor)
    assert out[3].asnumpy() == 4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_list_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        x.append(Tensor([5, 6]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))
    assert isinstance(out[2], Tensor)
    assert np.allclose(out[2].asnumpy(), np.array([5, 6]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_builtin_function_list_with_non_constant_tensor():
    """
    Feature: Graph list function.
    Description: When the input to list() is non constant tensor, list function will return correct result.
    Expectation: No exception.
    """
    @ms_function
    def foo(x):
        return list(x)

    ret = foo(Tensor([[1, 2, 3], [4, 5, 6]]))
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(ret[1].asnumpy() == np.array([4, 5, 6]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_tuple_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = tuple(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_tuple_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        return x
    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_builtin_function_tuple_with_non_constant_tensor():
    """
    Feature: Graph tuple function.
    Description: When the input to tuple() is non constant tensor, list function will return correct result.
    Expectation: No exception.
    """
    @ms_function
    def foo(x):
        return tuple(x)

    ret = foo(Tensor([[1, 2, 3], [4, 5, 6]]))
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(ret[1].asnumpy() == np.array([4, 5, 6]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_bool_with_input_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with tensor input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = Tensor([10])
        return bool(x)

    assert foo()
