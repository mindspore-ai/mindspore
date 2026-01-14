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
"""pynative function"""
import pytest
import numpy as np
import torch
from torch import nn as nn_pt
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops.operations as P
from mindspore.graph.api import jit
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark
from .utils import allclose_nparray
from .utils import GradOfAllInputs, GradOfFirstInput


class OpsFactory:
    def __init__(self, dtype=np.float16):
        super().__init__()
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype == np.float32:
            self.loss = 1e-4
        elif self.dtype == np.float64:
            self.loss = 1e-5
        else:
            self.loss = 0


class ParserFactory(OpsFactory):
    def __init__(self, net_me, net_torch, *init_input):
        super().__init__()
        self._input_num = len(init_input)
        self.net_me = net_me
        self.net_me.set_grad()

        self.net_torch = net_torch

        self.input_me_list = []
        self.input_pt_list = []

        for item in init_input:
            self._input_me = Tensor(item, ms.float32)
            self.input_me_list.append(self._input_me)

            if isinstance(item, int):
                self._input_pt = torch.tensor(item)
            else:
                self._input_pt = torch.from_numpy(item)
                self._input_pt.requires_grad = True

            self.input_pt_list.append(self._input_pt)

        self.out_np_shape = self.forward_mindspore_impl().shape
        if not self.out_np_shape:
            self.out_np = np.array(1).astype(np.float32)
        else:
            self.out_np = np.random.randn(*self.out_np_shape).astype(np.float32)
        self.output_grad_me = Tensor(self.out_np)
        self.output_grad_pt = torch.from_numpy(self.out_np)

    def forward_mindspore_impl(self):
        output_me = self.net_me(*self.input_me_list)
        return output_me

    def forward_pytorch_impl(self):
        output_pt = self.net_torch(*self.input_pt_list)
        return output_pt

    def grad_mindspore_impl(self):
        grad_func = GradOfFirstInput if self._input_num == 1 else GradOfAllInputs
        grad_net = grad_func(self.net_me)
        grad_net.set_train()
        grad_ms = grad_net(*self.input_me_list, self.output_grad_me)
        return grad_ms

    def grad_pytorch_impl(self):
        output_grad = torch.from_numpy(self.out_np.copy())
        out = self.net_torch(*self.input_pt_list)
        out.backward(gradient=output_grad)
        grad_pt = []
        for item in self.input_pt_list:
            grad_pt.append(item.grad)
        return grad_pt

    def forward_cmp(self):
        out_pytorch = self.forward_pytorch_impl().detach().numpy()
        out_mindspore = self.forward_mindspore_impl().asnumpy()
        allclose_nparray(out_pytorch, out_mindspore, self.loss, self.loss)

    def backward_cmp(self):
        grad_pt = self.grad_pytorch_impl()
        grad_ms = self.grad_mindspore_impl()
        for i in range(self._input_num):
            if grad_pt[i] is None:
                continue
            _grad_ms = grad_ms if self._input_num == 1 else grad_ms[i]
            input_grad_mindspore = _grad_ms.asnumpy()

            input_grad_pytorch = grad_pt[i].numpy()
            allclose_nparray(input_grad_pytorch, input_grad_mindspore, self.loss, self.loss)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_pynative_ms_function_control_for_multi_if_break(mode):
    """
    Feature: pynative function.
    Description: execute same function in mindspore and pytorch.
    Expectation: the result is same.
    """
    ms.set_context(mode=mode)

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.TensorAdd()

        @jit(backend="ms_backend")
        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 2 * x < y:
                    if 3 * x < y:
                        out = self.add(out, out)
                        x = x + 1
                    out = self.relu(out)
                if x + 6 == y:
                    break
            out = self.relu(out)
            return out

    class NetPytorch(nn_pt.Module):
        def __init__(self):
            super().__init__()
            self.range_num = 5

        def forward(self, x, y, z):
            out = z
            for _ in range(self.range_num):
                if 2 * x < y:
                    if 3 * x < y:
                        out = torch.add(out, out)
                        x = x + 1
                    out = torch.relu(out)
                if x + 6 == y:
                    break
            out = torch.relu(out)
            return out

    net = Net()
    net_pt = NetPytorch()
    fact = ParserFactory(net, net_pt, 2, 10, np.random.rand(4, 4, 4))
    fact.forward_cmp()
    fact.backward_cmp()
