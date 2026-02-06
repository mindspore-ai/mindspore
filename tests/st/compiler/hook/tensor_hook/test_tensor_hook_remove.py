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
"""Test remove operation of tensor hook."""

import numpy as np
import mindspore as ms
from mindspore.common import Tensor, Parameter, dtype, nn, ops
from mindspore.nn import Cell
from mindspore.train.serialization import export, load
import torch
import pytest
from tests.mark_utils import arg_mark


def double_fn(grad):
    return grad * 2


class Net5(Cell):
    def __init__(self):
        super().__init__()
        self.a = Parameter(Tensor(np.ones([2, 3], np.float32)), name='a')

    def construct(self, x):
        handle = x.register_hook(double_fn)
        out = x * x * self.a
        return out, handle


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_hook_register_in_remove_out():
    """
    Feature: Tensor hook.
    Description: Register a tensor hook inside @ms.jit and return the handle for removal outside.
    Expectation: Hook is correctly registered; returned handle is a Tensor.
    """
    input_np = np.ones([2, 3]) * 2
    input_x = Tensor(input_np, dtype.float32)
    input_x.register_hook(double_fn)
    net = Net5()
    _, handle = ms.jit(net)(input_x)
    grad_net = ops.grad(net, grad_position=0)
    ms.jit(grad_net)(input_x)
    assert isinstance(handle, Tensor)


class Net6(Cell):
    def __init__(self, hx):
        super().__init__()
        self.a = Parameter(Tensor(np.ones([2, 3], np.float32)), name='a')
        self.handle_a = self.a.register_hook(double_fn)
        self.handle_x = hx

    def construct(self, x):
        self.handle_x.remove()
        out = x * x * self.a
        return out


class Tet6(torch.nn.Module):
    def __init__(self, hx):
        super().__init__()
        self.a = torch.nn.parameter.Parameter(
            torch.tensor(np.ones([2, 3], np.float32)))
        self.handle_a = self.a.register_hook(double_fn)
        self.handle_x = hx

    def forward(self, x):
        self.handle_x.remove()
        out = x * x * self.a
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parse_hook_register_out_remove_in():
    """
    Feature: Tensor hook.
    Description: Register hook outside @ms.jit and attempt to remove it inside construct.
    Expectation: RuntimeError is raised because hook removal in construct is not supported.
    """
    input_np = np.ones([2, 3])
    input_x = Tensor(input_np, dtype.float32)
    handle_ms = input_x.register_hook(double_fn)
    net = Net6(handle_ms)

    with pytest.raises(RuntimeError) as ex:
        ms.jit(net)(input_x)
    assert "is not supported in 'construct'" in str(ex.value)


class Net7(Cell):
    def __init__(self):
        super().__init__()
        self.a = Parameter(Tensor(np.ones([2, 3], np.float32)), name='a')
        self.handle_a = self.a.register_hook(double_fn)

    def construct(self, x):
        out = x * x * self.a
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_parse_hook_register_export_mindir():
    """
    Feature: Tensor hook.
    Description: Register hook on input tensor, run forward and grad, then export to MINDIR and load for inference.
    Expectation: Gradients from original and loaded graph are numerically close.
    """
    input_np = np.ones([2, 3])
    input_x = Tensor(input_np, dtype.float32)
    input_x.register_hook(double_fn)
    net = Net7()
    ms.jit(net)(input_x)
    grad_net = ops.grad(net, grad_position=0, weights=net.a)
    grad_ms = ms.jit(grad_net)(input_x)

    export(grad_net, input_x, file_name='hook.mindir', file_format="MINDIR")
    infer_ms = nn.GraphCell(load('hook.mindir'))(input_x)
    np.allclose(grad_ms[0].asnumpy(), infer_ms[0].asnumpy(), 0.001, 0.001)
    np.allclose(grad_ms[1].asnumpy(), infer_ms[1].asnumpy(), 0.001, 0.001)


class Net8(Cell):
    def __init__(self):
        super().__init__()
        self.a = Parameter(Tensor(np.ones([2, 3], np.float32)), name='a')

    def construct(self, x):
        out = x * x * self.a
        return out

    def bprop(self, x, out, dout):
        return (x + out + dout,)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_parse_hook_bprop():
    """
    Feature: Tensor hook.
    Description: Register hook on input tensor and use custom bprop; verify gradient with hook effect.
    Expectation: Computed gradient matches expected value (input * 3) within tolerance.
    """
    input_np = np.ones([2, 3])
    input_x = Tensor(input_np, dtype.float32)
    input_x.register_hook(double_fn)
    net = Net8()
    ms.jit(net)(input_x)
    grad_net = ops.grad(net, grad_position=0)
    grad_ms = ms.jit(grad_net)(input_x)

    expect = input_np * 3
    np.allclose(expect, grad_ms.asnumpy(), 0.001, 0.001)
