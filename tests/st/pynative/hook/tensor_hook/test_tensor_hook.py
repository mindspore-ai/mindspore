# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
""" test_tensor_hook """
import os
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, Parameter, ops, nn
from tests.st.pynative.utils import GradOfAllParams, GradOfFirstInput, GradOfAllInputs
from tests.mark_utils import arg_mark


def hook_fn_mul_2(grad):
    return grad * 2


def hook_fn_mul_3(grad):
    return grad * 3


def hook_fn_return_tuple(grad):
    return grad, grad


def hook_fn_print_and_return_self(grad):
    print(grad.asnumpy())
    return grad


def test_tensor_backward_hook_with_op_output():
    """
    Feature: Tensor backward hook.
    Description: Test hook function on operation output.
    Expectation: Success
    """

    def hook_test(x, y):
        z = x * y
        z.register_hook(hook_fn_mul_2)
        z = z * y
        return z

    x = ms.Tensor([1.0, 2.0, 3.0], ms.float32)
    y = ms.Tensor([1.0, 2.0, 3.0], ms.float32)
    grad_x, grad_y = GradOfAllInputs(hook_test, sens_param=False)(x, y)

    assert np.allclose(grad_x.asnumpy(), np.array([2, 8, 18], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), np.array([3, 12, 27], dtype=np.float32), 0.00001, 0.00001)


def test_tensor_backward_hook_with_op_output_register_multi():
    """
    Feature: Tensor backward hook.
    Description: Test registering multiple hook on operation output.
    Expectation: Success
    """

    def hook_test_multi(x, y):
        z = x * y
        z.register_hook(hook_fn_mul_2)
        z.register_hook(hook_fn_mul_3)
        z = z * y
        return z

    x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    grad_x, grad_y = GradOfAllInputs(hook_test_multi, sens_param=False)(x, y)

    assert np.allclose(grad_x.asnumpy(), np.array([6, 24, 54], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), np.array([7, 28, 63], dtype=np.float32), 0.00001, 0.00001)


def hook_test_input(x, is_hook_inner=False):
    if is_hook_inner:
        x.register_hook(hook_fn_mul_2)
    y1 = x ** 2
    y2 = x + 1
    return y1 + y2


def test_tensor_backward_hook_with_net_input():
    """
    Feature: Tensor backward hook.
    Description: Test registering a backward hook on the input tensor, both outside and inside the network.
    Expectation: Success
    """
    x = ms.Tensor(1.0, ms.float32)
    handle = x.register_hook(hook_fn_mul_2)
    grad_x_hook_outer = GradOfFirstInput(hook_test_input, sens_param=False)(x, False)
    assert np.allclose(grad_x_hook_outer.asnumpy(), np.array([6], dtype=np.float32), 0.00001, 0.00001)

    handle.remove()
    grad_x_hook_inner = GradOfFirstInput(hook_test_input, sens_param=False)(x, True)
    assert np.allclose(grad_x_hook_inner.asnumpy(), grad_x_hook_outer.asnumpy(), 0.00001, 0.00001)


def test_tensor_backward_hook_with_net_input_register_multi():
    """
    Feature: Tensor backward hook.
    Description: Test registering multiple backward hook on network input.
    Expectation: Success
    """
    x = ms.Tensor(1.0, ms.float32)
    handle1 = x.register_hook(hook_fn_mul_2)
    handle2 = x.register_hook(hook_fn_mul_3)
    ms_grad = GradOfFirstInput(hook_test_input, False)
    grad_x = ms_grad(x)
    assert np.allclose(grad_x.asnumpy(), np.array([18], dtype=np.float32), 0.001, 0.001)

    handle1.remove()
    ms_grad = GradOfFirstInput(hook_test_input, False)
    grad_x = ms_grad(x)
    assert np.allclose(grad_x.asnumpy(), np.array([9], dtype=np.float32), 0.001, 0.001)

    handle1.remove()
    handle2.remove()
    ms_grad = GradOfFirstInput(hook_test_input, False)
    grad_x = ms_grad(x)
    assert np.allclose(grad_x.asnumpy(), np.array([3], dtype=np.float32), 0.001, 0.001)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight1")
        self.weight2 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight2")
        self.handle1 = self.weight1.register_hook(hook_fn_mul_2)
        self.handle2 = self.weight2.register_hook(hook_fn_mul_2)

    def construct(self, x):
        y = x * self.weight1
        z = x * self.weight2
        return y + z


def test_tensor_backward_hook_with_weight():
    """
    Feature: Tensor backward hook.
    Description: Test that backward hooks on network weights remain effective across multiple forward passes.
    Expectation: Success
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    net1 = Net()
    ms_grad = GradOfAllParams(net1, False)
    # First step
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), np.array([2, 4, 6], dtype=np.float32), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), np.array([2, 4, 6], dtype=np.float32), 0.001, 0.001)

    # Second step, no need register hook again
    input_x = Tensor(np.array([2.0, 3.0, 4.0]), ms.float32)
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), np.array([4, 6, 8], dtype=np.float32), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), np.array([4, 6, 8], dtype=np.float32), 0.001, 0.001)


def test_tensor_backward_hook_with_weight_register_multi():
    """
    Feature: Tensor backward hook.
    Description: Test multiple hook functions registration, removal, and cumulative effects on network weights.
    Expectation: Success
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    net1 = Net()
    ms_grad = GradOfAllParams(net1, False)
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), np.array([2, 4, 6], dtype=np.float32), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), np.array([2, 4, 6], dtype=np.float32), 0.001, 0.001)

    # Add multi hook fn
    handle1 = net1.weight1.register_hook(hook_fn_mul_3)
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), np.array([6, 12, 18], dtype=np.float32), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), np.array([2, 4, 6], dtype=np.float32), 0.001, 0.001)

    # Remove original hook fn
    net1.handle1.remove()
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), np.array([3, 6, 9], dtype=np.float32), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), np.array([2, 4, 6], dtype=np.float32), 0.001, 0.001)

    # remove all hook fn
    net1.handle1.remove()
    net1.handle2.remove()
    handle1.remove()
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), np.array([1, 2, 3], dtype=np.float32), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), np.array([1, 2, 3], dtype=np.float32), 0.001, 0.001)


class NetRemove(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight1")
        self.handle = self.weight1.register_hook(hook_fn_mul_2)

    def construct(self, x):
        x = x * self.weight1
        return x


def test_tensor_backward_hook_handle_remove():
    """
    Feature: Tensor backward hook.
    Description: Test hook removal functionality and verify gradient behavior before and after removal.
    Expectation: Success
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    net_remove = NetRemove()
    ms_grad = GradOfAllParams(net_remove, False)

    for i in range(2):
        if i == 0:
            output = ms_grad(input_x)
            assert np.allclose(output[0].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001,
                               0.001)
        else:
            net_remove.handle.remove()
            output = ms_grad(input_x)
            assert np.allclose(output[0].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001,
                               0.001)


not_in_grad = 1


def tensor_hook_fn2(grad):
    global not_in_grad
    not_in_grad += 1


class NetWithParameterNotInGrad(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight1")
        self.weight2 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight2")
        self.handle1 = self.weight1.register_hook(tensor_hook_fn2)
        self.handle2 = self.weight2.register_hook(tensor_hook_fn2)

    def construct(self, x):
        y = x * self.weight1
        y = y * self.weight1
        return y


def test_tensor_backward_hook_with_weight_not_in_grad():
    """
    Feature: Tensor hook
    Description: Test tensor hook for weight not in grad.
    Expectation: Success
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    net1 = NetWithParameterNotInGrad()
    ms_grad = GradOfAllParams(net1, False)
    ms_grad(input_x)
    assert not_in_grad == 3


def test_tensor_backward_hook_multi_output():
    """
    Feature: Tensor hook
    Description: Test tensor hook for multi output ops.
    Expectation: Success
    """

    def fn(x):
        y1, y2 = ops.split(x, 2)

        y1.register_hook(hook_fn_mul_2)
        handle = y1.register_hook(hook_fn_mul_3)

        y2.register_hook(hook_fn_mul_3)
        assert len(y1.hooks()) == 2
        assert len(y2.hooks()) == 1

        handle.remove()
        assert len(y1.hooks()) == 1
        return y1 + y2

    input_x = Tensor(np.arange(4).astype("float32"), dtype=ms.float32)
    grad_op = GradOfFirstInput(fn, sens_param=False)
    grad = grad_op(input_x)
    assert np.allclose(grad.asnumpy(), np.array([2.0, 2.0, 3.0, 3.0], dtype=np.float32), 0.001, 0.001)


def test_tensor_backward_hook_leaf():
    """
    Feature: Tensor hook
    Description: Register tensor hook for leaf node.
    Expectation: Success
    """

    def fn(x):
        y = x * x
        z = ops.relu(y)
        return z

    input_x = Tensor([1.0, 2.0], dtype=ms.float32)
    handle = input_x.register_hook(hook_fn_mul_2)
    assert len(input_x.hooks()) == 1

    grad_op = GradOfFirstInput(fn, sens_param=False)
    grad = grad_op(input_x)
    assert np.allclose(grad.asnumpy(), np.array([4.0, 8.0], dtype=np.float32), 0.001, 0.001)

    assert len(input_x.hooks()) == 1
    handle.remove()
    assert not input_x.hooks()


def test_tensor_backward_hook_return_none():
    """
    Feature: Tensor backward hook.
    Description: Test hook return None.
    Expectation: Not change gradient.
    """

    record = 0

    def hook_record(unused):
        nonlocal record
        record = 1

    def fn(x):
        x.register_hook(hook_record)
        return x + 1.0

    x = ms.Tensor(2.0)
    grad_x = GradOfFirstInput(fn, sens_param=False)(x)
    assert np.allclose(grad_x.asnumpy(), np.array([1.0], dtype=np.float32), 0.001, 0.001)
    assert record == 1


def test_tensor_backward_hook_print_and_return_self():
    """
    Feature: Tensor backward hook.
    Description: Test hook print and return self.
    Expectation: Success.
    """

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.n = 2

        def construct(self, x):
            return x * self.n

    net = Net1()
    x = ms.Tensor(np.random.rand(3, ), dtype=ms.float32)
    x.register_hook(hook_fn_print_and_return_self)
    grad_net = GradOfFirstInput(net, sens_param=False)
    grad = grad_net(x)
    assert np.allclose(grad.asnumpy(), np.array([2.0, 2.0, 2.0]), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook():
    """
    Feature: Tensor backward hook.
    Description: Test suite for tensor hook.
    Expectation: Success
    """
    test_tensor_backward_hook_with_op_output()
    test_tensor_backward_hook_with_op_output_register_multi()
    test_tensor_backward_hook_with_net_input()
    test_tensor_backward_hook_with_net_input_register_multi()
    test_tensor_backward_hook_with_weight()
    test_tensor_backward_hook_with_weight_register_multi()
    test_tensor_backward_hook_handle_remove()
    test_tensor_backward_hook_with_weight_not_in_grad()
    test_tensor_backward_hook_multi_output()
    test_tensor_backward_hook_leaf()
    test_tensor_backward_hook_return_none()
    test_tensor_backward_hook_print_and_return_self()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_leaf_error():
    """
    Feature: Tensor backward hook.
    Description: Register tensor hook for parameters that do not require gradients.
    Expectation: Raise RuntimeError.
    """
    match_error_str = "The tensor requires grad is false, which can not register tensor hook"

    param_x = Parameter(Tensor(np.array([2.0, 3.0]), ms.float32), requires_grad=False)
    with pytest.raises(RuntimeError, match=match_error_str):
        param_x.register_hook(hook_fn_mul_2)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_return_non_tensor():
    """
    Feature: Tensor backward hook.
    Description: Test hook return non-tensor.
    Expectation: Raise RuntimeError.
    """

    def fn(x):
        x.register_hook(hook_fn_return_tuple)
        return x * x

    x = ms.Tensor(1.0)
    with pytest.raises(RuntimeError, match="Tensor hook should be return Tensor, but get type"):
        GradOfFirstInput(fn, sens_param=False)(x)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_register_error():
    """
    Feature: Tensor backward hook.
    Description: Test registering non-callable hook.
    Expectation: Raise TypeError.
    """
    x = ms.Tensor([1.0])
    with pytest.raises(TypeError, match="Expected a callable hook function"):
        x.register_hook(1.0)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_tensor_backward_hook_inplace():
    """
    Feature: Tensor backward hook.
    Description: Test tensor hook behavior with inplace operations.
    Expectation: Success
    """

    def fn(x):
        y = x * x
        y.register_hook(hook_fn_mul_2)
        y.add_(2.0)
        assert not y.hooks()
        y.register_hook(hook_fn_mul_3)
        assert len(y.hooks()) == 1
        return y + 1.0

    input_x = Tensor([1.0, 2.0], dtype=ms.float32)
    grad_op = GradOfFirstInput(fn, sens_param=False)
    grad = grad_op(input_x)
    assert np.allclose(grad.asnumpy(), np.array([12.0, 24.0], dtype=np.float32), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='allcards',
          essential_mark='essential')
def test_tensor_hook_with_reduce_scatter():
    """
    Feature: mpi run 8P case of 'reduce_scatter' communication operator for pynative tensor hook.
    Description: mpi run 8P case of 'reduce_scatter' communication operator for pynative tensor hook.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_tensor_hook_reduce_scatter.py")
    assert return_code == 0


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_with_jit():
    """
    Feature: Tensor hook
    Description: register a tensor hook decorated with jit.
    Expectation: Raise TypeError.
    """

    @ms.jit
    def hook_fn_with_jit(grad):
        return grad * 2.0

    x = ms.Tensor(1.0)
    with pytest.raises(TypeError):
        x.register_hook(hook_fn_with_jit)
