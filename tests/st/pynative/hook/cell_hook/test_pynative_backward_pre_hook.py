# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import jit, ops
from mindspore.ops import GradOperation
from tests.mark_utils import arg_mark
from tests.st.pynative.hook.cell_hook.common import assert_jit_grad_net_by_grad_op

context.set_context(mode=context.PYNATIVE_MODE)


def backward_pre_hook_fn_with_no_return(cell, grad_outp):
    print("get grad output ", grad_outp)


def backward_pre_hook_fn_with_old_return(cell, grad_outp):
    return grad_outp


def backward_pre_hook_fn_with_new_return(cell, grad_outp):
    return grad_outp[0] * 3


def backward_pre_hook_modify_cell(cell, grad_outp):
    print("cell.a ", cell.a)
    cell.a = 2
    print("get grad output ", grad_outp)
    return grad_outp[0] * 2


def backward_hook_with_new_return(cell, grad_inp, grad_outp):
    return grad_inp[0] * 4, grad_inp[1] * 4


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.mul.a = 1

    def construct(self, x, y):
        x = self.mul(x, y)
        x = x + x
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('hook', [backward_pre_hook_fn_with_no_return, backward_pre_hook_fn_with_old_return])
def test_pynative_backward_pre_hook_on_net(hook):
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function (reg hook for net).
    Expectation: The calculation result is correct.
    """

    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    net.register_backward_pre_hook(hook)
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    expect_out = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)

    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)


def test_pynative_backward_pre_hook_with_no_return():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function (reg hook for net.mul).
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_no_return)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_out = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)


def test_pynative_backward_pre_hook_with_old_return():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function.
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_old_return)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_out = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)


def test_pynative_backward_pre_hook_with_new_return():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function.
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_out = Tensor(np.ones([1]).astype(np.float32) * 6)
    assert np.allclose(grad[0].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)


def test_pynative_backward_pre_hook_with_modify_cell():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function.
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    net.mul.register_backward_pre_hook(backward_pre_hook_modify_cell)
    grad = grad_op(net)(input_x, input_y)
    # hook with memory side effect is not supported now
    # assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    assert net.mul.a == 2
    expect_out = Tensor(np.ones([1]).astype(np.float32) * 4)
    assert np.allclose(grad[0].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)


def test_pynative_backward_pre_hook_with_new_return_multi_register():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function.
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_out = Tensor(np.ones([1]).astype(np.float32) * 18)
    assert np.allclose(grad[0].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)


def test_pynative_backward_pre_hook_with_handle_remove():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function.
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    # Step1: register hook
    handle1 = net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    handle2 = net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_out = Tensor(np.ones([1]).astype(np.float32) * 18)
    assert np.allclose(grad[0].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)

    # Step2: remove hook by handle1
    handle1.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 6)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)

    # Step3: remove hook by handle2
    handle2.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)


def test_pynative_backward_pre_hook_with_backward_hook_and_remove():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function.
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    # Step1: register hook
    handle1 = net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    handle2 = net.mul.register_backward_hook(backward_hook_with_new_return)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 24)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)

    # Step2: remove hook by handle1, backward hook needs work
    handle1.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 8)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)

    # Step3: remove hook by handle2, no hook work
    handle2.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)


def test_pynative_backward_pre_hook_with_backward_hook_multi_register():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward pre hook function.
    Expectation: The calculation result is correct.
    """
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = Net()
    # Step1: register hook
    handle1 = net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    handle2 = net.mul.register_backward_pre_hook(backward_pre_hook_fn_with_new_return)
    handle3 = net.mul.register_backward_hook(backward_hook_with_new_return)
    handle4 = net.mul.register_backward_hook(backward_hook_with_new_return)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 288)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)

    # Step2: remove backward pre hook by handle2
    handle2.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 96)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)

    # Step3: remove hook by handle3 and handle4, no hook work
    handle4.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 24)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)

    # Step4: remove all hooks
    handle1.remove()
    handle3.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_cell_backward_pre_hook_test_suite():
    """
    Feature: PyNative cell backward hook.
    Description: Test suite for pynative cell backward pre hook.
    Expectation: Success
    """
    test_pynative_backward_pre_hook_with_no_return()
    test_pynative_backward_pre_hook_with_old_return()
    test_pynative_backward_pre_hook_with_new_return()
    test_pynative_backward_pre_hook_with_modify_cell()
    test_pynative_backward_pre_hook_with_new_return_multi_register()
    test_pynative_backward_pre_hook_with_handle_remove()
    test_pynative_backward_pre_hook_with_backward_hook_and_remove()
    test_pynative_backward_pre_hook_with_backward_hook_multi_register()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_backward_pre_hook_with_wrong_number_of_output():
    """
    Feature: PyNative hook function with jit.
    Description: Test PyNative hook function with jit.
    Expectation: Raise error.
    """

    class GradOfAllInputs(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=True)

        @jit
        def construct(self, *inputs):
            grad_net = self.grad_op(self.net)
            return grad_net(*inputs)

    class InnerNet(nn.Cell):
        def __init__(self):
            super(InnerNet, self).__init__()
            self.mul = MulNet()
            self.handle = self.mul.register_backward_pre_hook(double_pback)

        def construct(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    class MulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = ops.Mul()
            self.relu = nn.ReLU()

        def construct(self, x, y):
            x = self.mul(x, y)
            x = self.relu(x)
            return x

    def double_pback(cell, grad_output):
        return grad_output * 2

    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)
    ms_net = InnerNet()
    grad_net = GradOfAllInputs(ms_net)
    with pytest.raises(TypeError) as err:
        grad_net(input1_ms, input2_ms)
        assert "The backward pre hook return value size is" in str(err.value)
