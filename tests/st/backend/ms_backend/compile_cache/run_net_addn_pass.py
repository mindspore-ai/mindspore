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
"""
test compile cache with kernel packet reducesum.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import compare_nparray
from tests.st.pynative.utils import GradOfAllInputs

import numpy as np
import torch
import torch.nn as nn_pt
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.graph.api import jit


class ParserFactory():
    """
    Run net, get and compare forward and backward output with input.
    """
    def __init__(self, net_ms, net_torch, *init_input):
        super().__init__()
        self._input_num = len(init_input)
        self.net_ms = net_ms
        self.net_ms.set_grad()

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

        output_me = self.net_ms(*self.input_me_list)
        self.out_np_shape = output_me.shape
        if not self.out_np_shape:
            self.out_np = np.array(1).astype(np.float32)
        else:
            self.out_np = np.random.randn(*self.out_np_shape).astype(np.float32)
        self.output_grad_ms = Tensor(self.out_np)
        self.output_grad_pt = torch.from_numpy(self.out_np)


    def forward_cmp(self):
        output_me = self.net_ms(*self.input_me_list)
        output_pt = self.net_torch(*self.input_pt_list)
        output_pt_np = output_pt.detach().numpy()
        output_ms_np = output_me.asnumpy()
        compare_nparray(output_pt_np, output_ms_np, 1e-3, 1e-3)
        return output_ms_np

    def backward_cmp(self):
        output_pt_grad = torch.from_numpy(self.out_np.copy())
        out_pt = self.net_torch(*self.input_pt_list)
        out_pt.backward(gradient=output_pt_grad)
        grad_pt = []
        for item in self.input_pt_list:
            grad_pt.append(item.grad)

        grad_net = GradOfAllInputs(self.net_ms)
        grad_net.set_train()
        grad_ms = grad_net(*self.input_me_list, self.output_grad_ms)

        for i in range(self._input_num):
            if grad_pt[i] is None:
                continue
            _grad_ms = grad_ms[i]
            input_grad_mindspore = _grad_ms.asnumpy()

            input_grad_pytorch = grad_pt[i].numpy()
            compare_nparray(input_grad_pytorch, input_grad_mindspore, 1e-3, 1e-3)


class Net(Cell):
    """
    Net MindSpore
    """
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
    """
    Net Pytorch
    """
    def forward(self, x, y, z):
        out = z
        for _ in range(5):
            if 2 * x < y:
                if 3 * x < y:
                    out = torch.add(out, out)
                    x = x + 1
                out = torch.relu(out)
            if x + 6 == y:
                break
        out = torch.relu(out)
        return out


if __name__ == "__main__":
    net = Net()
    net_pt = NetPytorch()
    fact = ParserFactory(net, net_pt, 2, 10, np.ones((4, 4, 4), np.float32))
    out_mindspore = fact.forward_cmp()
    fact.backward_cmp()
    print("RUNTIME_COMPILE", out_mindspore[0][0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", out_mindspore[0].shape, "RUNTIME_CACHE")
