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
""" Test tensor hook"""

import os
import subprocess
import shutil
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor, Parameter, context, nn, ops, dtype
from tests.mark_utils import arg_mark


def hook_double(grad):
    return grad * 2


def hook_triple(grad):
    return grad * 3


def hook_mul_5(grad):
    return grad * 5


def hook_print(grad):
    print("grad:", grad)


def hook_double_with_print(grad):
    print("hook_double")
    return grad * 2


def hook_with_ctrl_flow(grad):
    if grad[0] < 10000000:
        return hook_double(grad)
    return hook_triple(grad)


np_weight0 = np.array([1.0, 2.0, 3.0])
np_weight1 = np.array([4.0, 5.0, 6.0])
np_input_x = np.array([7.0, 8.0, 9.0])


class GroundNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        out = x * self.weight1
        return out


class OneTensorOneHookNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_double)
        out = x * self.weight1
        return out


class OneTensorMultiHookNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_double)
        x.register_hook(hook_triple)
        out = x * self.weight1
        return out


class MultiTensorMultiHookNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_double)
        x.register_hook(hook_triple)
        y = x * self.weight1
        y.register_hook(hook_double)
        y.register_hook(hook_triple)
        out = y
        return out


class HookPrintNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_print)
        out = x * self.weight1
        return out


class HookInJITNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    @ms.jit
    def hook(self, x):
        x.register_hook(hook_double_with_print)
        return x

    def construct(self, x):
        x = x * self.weight0
        x = self.hook(x)
        out = x * self.weight1
        return out


class CtrlFlowHookInJITNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    @ms.jit
    def hook(self, x):
        x.register_hook(hook_with_ctrl_flow)
        return x

    def construct(self, x):
        x = x * self.weight0
        x = self.hook(x)
        out = x * self.weight1
        return out


class NeedReorderHookStmtNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x.register_hook(hook_double)
        x = x * self.weight0 + x - x
        x.register_hook(hook_triple)
        x = x * self.weight1 - x + x
        x.register_hook(hook_mul_5)
        return x


ground_input_x = Tensor(np_input_x, ms.float32)
ground_net = GroundNet()
ground_grad_op = ops.GradOperation(get_all=True, get_by_list=True)
ground_grad_net = ground_grad_op(ground_net, ground_net.trainable_params())
ground_output = ground_grad_net(ground_input_x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_one_hook():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register one hook on one tensor.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    net = OneTensorOneHookNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_double(ground_output[0][0]).asnumpy()
    expected_weight0_grad = hook_double(ground_output[1][0]).asnumpy()
    expected_weight1_grad = ground_output[1][1].asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_multi_hook():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register multi hooks on one tensor.
    Expectation: The grad of the tensor is changed by hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    net = OneTensorMultiHookNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    with pytest.raises(RuntimeError) as e:
        grad_net(input_x)
        output = grad_net(input_x)
        output_x_grad = output[0][0].asnumpy()
        output_weight0_grad = output[1][0].asnumpy()
        output_weight1_grad = output[1][1].asnumpy()
        expected_x_grad = hook_double(
            hook_triple(ground_output[0][0])).asnumpy()
        expected_weight0_grad = hook_double(
            hook_triple(ground_output[1][0])).asnumpy()
        expected_weight1_grad = ground_output[1][1].asnumpy()

        assert np.allclose(output_x_grad, expected_x_grad)
        assert np.allclose(output_weight0_grad, expected_weight0_grad)
        assert np.allclose(output_weight1_grad, expected_weight1_grad)
    assert "It is not supported to register multiple hooks for a Tensor." in str(
        e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_tensor_multi_hook():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register multi hooks on multi tensors.
    Expectation: The grad of each tensor is changed by its registered hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    net = MultiTensorMultiHookNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    with pytest.raises(RuntimeError) as e:
        output = grad_net(input_x)
        output_x_grad = output[0][0].asnumpy()
        output_weight0_grad = output[1][0].asnumpy()
        output_weight1_grad = output[1][1].asnumpy()
        expected_x_grad = hook_double(hook_triple(
            hook_double(hook_triple(ground_output[0][0])))).asnumpy()
        expected_weight0_grad = hook_double(hook_triple(
            hook_double(hook_triple(ground_output[1][0])))).asnumpy()
        expected_weight1_grad = hook_double(
            hook_triple(ground_output[1][1])).asnumpy()

        assert np.allclose(output_x_grad, expected_x_grad)
        assert np.allclose(output_weight0_grad, expected_weight0_grad)
        assert np.allclose(output_weight1_grad, expected_weight1_grad)
    assert "It is not supported to register multiple hooks for a Tensor." in str(
        e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hook_no_return():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register no return hook on one tensor.
    Expectation: The grad of the tensor is not changed and the hook is applied(by check ir file).
    """
    save_graphs_path = "./test_tensor_hook_inside_graph_no_return"
    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=True, save_graphs_path=save_graphs_path)

    input_x = Tensor(np_input_x, ms.float32)
    net = HookPrintNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = ground_output[0][0].asnumpy()
    expected_weight0_grad = ground_output[1][0].asnumpy()
    expected_weight1_grad = ground_output[1][1].asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)

    para = 'Print("grad:'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" %
            (para, os.path.join(save_graphs_path, "*validate*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "1"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_need_reorder_hook_stmt_net():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook after tensor is used.
    Expectation: The grad of the tensor is changed (equal to reorder hook stmt before tensor is used).
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

    input_x = Tensor(np_input_x, ms.float32)
    net = NeedReorderHookStmtNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_mul_5(hook_triple(
        hook_double(ground_output[0][0]))).asnumpy()
    expected_weight0_grad = hook_mul_5(
        hook_triple(ground_output[1][0])).asnumpy()
    expected_weight1_grad = hook_mul_5(ground_output[1][1]).asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode, net', [
    (context.PYNATIVE_MODE, HookInJITNet()),
    (context.PYNATIVE_MODE, CtrlFlowHookInJITNet()),
    (context.GRAPH_MODE, HookInJITNet()),
    (context.GRAPH_MODE, CtrlFlowHookInJITNet())])
def test_hook_in_jit(mode, net):
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook inside jit wrapper.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_double(ground_output[0][0]).asnumpy()
    expected_weight0_grad = hook_double(ground_output[1][0]).asnumpy()
    expected_weight1_grad = ground_output[1][1].asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)


class Net5(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = Parameter(
            Tensor(np.ones([2, 3]), dtype=dtype.float32), name='a')
        self.k = 2

    def double_method(self, grad):
        return grad * self.k

    @ms.jit
    def construct(self, x):
        x.register_hook(self.double_method)
        out = x * x * self.a
        return out


class Tet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.parameter.Parameter(
            torch.tensor(np.ones([2, 3]), dtype=torch.float))
        self.k = 2

    def double_method(self, grad):
        return grad * self.k

    def forward(self, x):
        x.register_hook(self.double_method)
        out = x * x * self.a
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_parse_hook_register_method():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook of type method.
    Expectation: The grad of tensor is changed by hook.
    """
    input_np = np.ones([2, 3])
    input_x = Tensor(input_np, dtype.float32)
    net = Net5()
    grad_net = ops.grad(net)
    grad_ms = grad_net(input_x)

    tet = Tet5()
    input_pt = torch.tensor(input_np, dtype=torch.float)
    input_pt.requires_grad = True
    sens = torch.ones_like(input_pt)
    out_pt = tet(input_pt)
    out_pt.backward(sens)
    grad_pt = input_pt.grad

    np.allclose(grad_pt.detach().numpy(), grad_ms.asnumpy(), 0.001, 0.001)


def lam_hook(grad):
    return 2 * grad


class Net6(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = Parameter(
            Tensor(np.ones([2, 3]), dtype=dtype.float32), name='a')
        self.k = 2

    @ms.jit
    def construct(self, x):
        x.register_hook(lam_hook)
        out = x * x * self.a
        return out


class Tet6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.parameter.Parameter(
            torch.tensor(np.ones([2, 3]), dtype=torch.float))
        self.k = 2

    def forward(self, x):
        x.register_hook(lambda grad: 2 * grad)
        out = x * x * self.a
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_parse_hook_register_lambda():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook of type lambda.
    Expectation: The grad of tensor is changed by hook.
    """
    input_np = np.ones([2, 3])
    input_x = Tensor(input_np, dtype.float32)
    net = Net6()
    grad_net = ops.grad(net)
    grad_ms = grad_net(input_x)

    tet = Tet6()
    input_pt = torch.tensor(input_np, dtype=torch.float)
    input_pt.requires_grad = True
    sens = torch.ones_like(input_pt)
    out_pt = tet(input_pt)
    out_pt.backward(sens)
    grad_pt = input_pt.grad

    np.allclose(grad_pt.detach().numpy(), grad_ms.asnumpy(), 0.001, 0.001)


def square_np(grad):
    grad_np = grad.asnumpy()
    out_np = np.square(grad_np)
    return Tensor(out_np)


def square_pt(grad):
    grad_np = grad.detach().numpy()
    out_np = np.square(grad_np)
    return torch.tensor(out_np)


class Net8(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = Parameter(
            Tensor(np.ones([2, 3]), dtype=dtype.float32), name='a')
        self.k = 2

    @ms.jit
    def construct(self, x):
        out = x * x * self.a
        return out


class Tet8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.parameter.Parameter(
            torch.tensor(np.ones([2, 3]), dtype=torch.float))
        self.k = 2

    def forward(self, x):
        out = x * x * self.a
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_parse_hook_register_fallback():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook with fallback in hook.
    Expectation: The grad of tensor is changed by hook.
    """
    input_np = np.ones([2, 3])
    input_x = Tensor(input_np, dtype.float32)
    input_x.register_hook(square_np)
    net = Net8()
    grad_net = ops.grad(net)
    grad_ms = grad_net(input_x)

    tet = Tet8()
    input_pt = torch.tensor(input_np, dtype=torch.float)
    input_pt.requires_grad = True
    input_pt.register_hook(square_pt)
    sens = torch.ones_like(input_pt)
    out_pt = tet(input_pt)
    out_pt.backward(sens)
    grad_pt = input_pt.grad

    np.allclose(grad_pt.detach().numpy(), grad_ms.asnumpy(), 0.001, 0.001)


def double_fn(grad):
    return 2 * grad


class Net12(nn.Cell):
    def __init__(self):
        super().__init__()
        self.k = 2

    def construct(self, x):
        y = Tensor(x.asnumpy())
        y.register_hook(double_fn)
        out = y * self.k
        return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_parse_hook_register_middle_fallback():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook on a Tensor that is created explicitly in jit.
    Expectation: The grad of tensor is changed by hook.
    """
    input_np = np.ones([2, 3])
    input_x = Tensor(input_np, dtype.float32)
    net = Net12()
    grad_net = ops.grad(net)
    grad_ms = ms.jit(grad_net, backend="ms_backend")(input_x)

    np.allclose(np.zeros([2, 3], np.float32), grad_ms.asnumpy(), 0.001, 0.001)
