# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test cell hook"""

import pytest
import numpy as np
import torch
from torch.nn import Module
import mindspore as ms
from mindspore import nn, Tensor, Parameter, ops
from tests.mark_utils import arg_mark


class MulNetTorch(Module):
    def __init__(self):
        super().__init__()
        self.mul = torch.mul
        self.relu = torch.nn.ReLU()

    def forward(self, x, y):
        x = self.mul(x, y)
        x = self.relu(x)
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


class GradOfAllInputs(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, *inputs):
        grad_net = self.grad_op(self.net)
        return grad_net(*inputs)


def double_fn(cell_id, inputs):
    modified_inputs = (inputs[0] * 2, inputs[1] * 2)
    return modified_inputs


def double_back(cell, grad_input, grad_output):
    return grad_input[0] + grad_output[0], grad_input[1] + grad_output[0]


def double_pback(cell, grad_output):
    return tuple(g * 2 for g in grad_output)


def forward_hook_fn(cell, inputs, output):
    return output + inputs[0] * inputs[1]


def compare_with_torch(ms_net, torch_net):
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)

    ms_net.set_grad()
    out_ms = ms_net(input1_ms, input2_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input1_ms, input2_ms, out_ms)

    input1_torch = torch.from_numpy(input1_np)
    input2_torch = torch.from_numpy(input2_np)
    input1_torch.requires_grad = True
    input2_torch.requires_grad = True

    out_torch = torch_net(input1_torch, input2_torch)
    out_torch.backward(out_torch)

    assert np.allclose(out_torch.detach().numpy(),
                       out_ms.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(input1_torch.grad.numpy(),
                       input_ms_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(input2_torch.grad.numpy(),
                       input_ms_grad[1].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_jit_cell_hook(mode):
    """
    Feature: Cell Hook.
    Description: Test cell hook with jit.
    Expectation: No exception.
    """
    class Mod(Module):
        def __init__(self):
            super().__init__()
            self.mul = MulNetTorch()
            self.handle71 = self.mul.register_forward_pre_hook(double_fn)
            self.handle72 = self.mul.register_forward_hook(forward_hook_fn)
            self.handle73 = self.mul.register_full_backward_pre_hook(
                double_pback)
            self.handle74 = self.mul.register_full_backward_hook(double_back)

        def forward(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = MulNet()
            self.handle75 = self.mul.register_forward_pre_hook(double_fn)
            self.handle76 = self.mul.register_forward_hook(forward_hook_fn)
            self.handle77 = self.mul.register_backward_pre_hook(double_pback)
            self.handle78 = self.mul.register_backward_hook(double_back)

        def construct(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    ms.set_context(mode=mode)
    ms_net = Net()
    torch_net = Mod()
    compare_with_torch(ms_net, torch_net)


def print_fn(cell, inputs, outputs):
    print(inputs)
    print(outputs)


class MulNetTorch1(Module):
    def forward(self, x, y):
        return x[0] * y['a']


class MulNet1(nn.Cell):
    def construct(self, x, y):
        return x[0] * y['a']


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_cell_hook_with_tuple_list_dict_as_input():
    """
    Feature: Cell Hook.
    Description: Test cell hook with jit with tuple/list/dict as input.
    Expectation: Forward output matches PyTorch; hooks execute without error.
    """
    class Mod(Module):
        def __init__(self):
            super().__init__()
            self.mul = MulNetTorch1()
            self.handle = self.mul.register_forward_hook(print_fn)

        def forward(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = MulNet1()
            self.handle = self.mul.register_forward_hook(print_fn)

        def construct(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    ms_net = Net()
    torch_net = Mod()
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)
    x_ms = (input1_ms, input2_ms)
    y_ms = {'a': input1_ms}

    out_ms = ms.jit(ms_net)(x_ms, y_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    ms.jit(grad_net)(x_ms, y_ms, out_ms)

    input1_torch = torch.from_numpy(input1_np)
    input2_torch = torch.from_numpy(input2_np)
    x_pt = (input1_torch, input2_torch)
    y_pt = {'a': input1_torch}

    out_torch = torch_net(x_pt, y_pt)

    assert np.allclose(out_torch.detach().numpy(),
                       out_ms.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_cell_hook_with_int_as_input():
    """
    Feature: Cell Hook.
    Description: Test cell hook with jit with int as input.
    Expectation: Forward and gradient outputs match PyTorch; hooks handle scalar inputs correctly.
    """
    def half_fn_1(cell, inputs, outputs):
        return inputs[0] / inputs[1]

    class Mod(Module):
        def __init__(self):
            super().__init__()
            self.mul = MulNetTorch()
            self.handle = self.mul.register_forward_hook(half_fn_1)

        def forward(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = MulNet()
            self.handle = self.mul.register_forward_hook(half_fn_1)

        def construct(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    ms_net = Net()
    torch_net = Mod()
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2 = 2

    input1_ms = Tensor(input1_np)
    ms_net.set_grad()
    out_ms = ms.jit(ms_net)(input1_ms, input2)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = ms.jit(grad_net)(input1_ms, input2, out_ms)

    input1_torch = torch.from_numpy(input1_np)
    input1_torch.requires_grad = True
    out_torch = torch_net(input1_torch, input2)
    out_torch.backward(out_torch)

    np.allclose(out_torch.detach().numpy(), out_ms.asnumpy(), 0.00001, 0.00001)
    np.allclose(input1_torch.grad.numpy(),
                input_ms_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_cell_hook_with_none_as_input():
    """
    Feature: Cell Hook.
    Description: Test cell hook with jit with None as input.
    Expectation: Hooks process inputs containing None without error; graph compiles and runs successfully.
    """
    def half_fn_1(cell, inputs, outputs):
        print("inputs[0]", inputs[0])
        print("inputs[1]", inputs[1])
        return inputs[0], inputs[1]

    class EyeLayer(nn.Cell):
        def construct(self, *args, **kwargs):
            return ops.eye(*args, **kwargs)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.eye = EyeLayer()
            self.handle = self.eye.register_forward_hook(half_fn_1)

        def construct(self, x, y):
            x = self.relu(x)
            x = self.eye(x.shape[0], y)
            return x, y

    ms_net = Net()
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2 = None

    input1_ms = Tensor(input1_np)
    ms_net.set_grad()
    out_ms = ms.jit(ms_net)(input1_ms, input2)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    ms.jit(grad_net)(input1_ms, input2, out_ms)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_cell_hook_with_kwargs_as_input():
    """
    Feature: Cell Hook.
    Description: Test cell hook with jit with kwargs as input.
    Expectation: A TypeError should raised.
    """
    def forward_hook_fn_2(cell, inputs, kwargs, output):
        print(f"Forward arguments: {kwargs}")
        if not inputs:
            x = kwargs['x']
            y = kwargs['y']
            return output + x * y
        return output + inputs[0] * inputs[1]

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = MulNet()
            self.handle = self.mul.register_forward_hook(
                forward_hook_fn_2, with_kwargs=True)

        def construct(self, x, y):
            x = x + x
            x = self.mul(x=x, y=y)
            return x

    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)

    ms_net = Net()

    with pytest.raises(TypeError):
        ms.jit(ms_net)(input1_ms, input2_ms)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_cell_hook_with_side_effect_in_hook():
    """
    Feature: Cell Hook.
    Description: Test cell hook with jit with side effect in hook.
    Expectation: A TypeError should raised.
    """
    def forward_hook_fn_3(cell, inputs, output):
        updated_param = cell.param + inputs[0]
        cell.param = Parameter(updated_param)
        return output + inputs[0] * inputs[1]

    class AssignCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign = ops.Assign()
            self.param = Parameter(Tensor([0], dtype=ms.int32))

        def construct(self, x, y):
            result = self.assign(x, y)
            return result, y

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.assign_cell = AssignCell()
            self.handle = self.assign_cell.register_forward_hook(
                forward_hook_fn_3)

        def construct(self, x, y):
            x, y = self.assign_cell(x, y * 2)
            return self.add(x * 2, y * 3)

    x = Parameter(Tensor([2], dtype=ms.int32))
    y = Tensor([3], dtype=ms.int32)
    net = Net()
    with pytest.raises(ValueError):
        ms.jit(net)(x, y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_cell_hook_reg_in_jit():
    """
    Feature: Cell Hook.
    Description: Test cell hook with jit and register hook in jit scope.
    Expectation: A TypeError should raised.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = MulNet()

        def construct(self, x, y):
            x = x + x
            self.mul.register_forward_pre_hook(double_fn)
            x = self.mul(x, y)
            return x

    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    ms_net = Net()
    ms_net.set_grad()

    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)

    with pytest.raises(RuntimeError) as e:
        ms.jit(ms_net)(input1_ms, input2_ms)
    assert "Failed to compile in GRAPH_MODE because" in str(e.value)
