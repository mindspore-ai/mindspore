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
test compile cache with dynamic shape net.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import compare_nparray

import numpy as np
from torch import tensor
from torch.nn import Module
from mindspore.nn import Cell
from mindspore.common import jit, Tensor, enable_dynamic, dtype



d = Tensor(shape=[None, None], dtype=dtype.float32)
class ShapeAdd(Cell):
    """ShapeAdd
        Args:
            None

        Inputs:
            x (Tensor): First input tensor of any numeric type.
            y (Tensor): Second input tensor of any numeric type.

        Returns:
            Tuple, the element-wise sum of the two tensors' shapes plus a constant tuple (1,).

        Examples:
            >>> x = ms.Tensor(np.ones((2, 3), np.float32))
            >>> y = ms.Tensor(np.ones((4, 5), np.float32))
            >>> net = ShapeAdd()
            >>> output = net(x, y)
    """
    def __init__(self):
        super().__init__()
        self.eps = (1,)

    @jit(backend="ms_backend")
    @enable_dynamic(ms_input_x=d, ms_input_y=d)
    def construct(self, ms_input_x, ms_input_y):
        return ms_input_x.shape + ms_input_y.shape + self.eps


class ShapeAddT(Module):
    """ShapeAddT
        Args:
            None

        Inputs:
            x (Tensor): First input tensor of any numeric type.
            y (Tensor): Second input tensor of any numeric type.

        Returns:
            Tuple, the element-wise sum of the two tensors' shapes plus a constant tuple (1,).

        Examples:
            >>> x = torch.tensor(np.ones((2, 3), np.float32))
            >>> y = torch.tensor(np.ones((4, 5), np.float32))
            >>> net = ShapeAddT()
            >>> output = net(x, y)
    """
    def __init__(self):
        super().__init__()
        self.eps = (1,)

    def forward(self, torch_input_x, torch_input_y):
        return torch_input_x.shape + torch_input_y.shape + self.eps


if __name__ == "__main__":
    input_x = np.ones([4, 6], np.float32)
    input_y = np.ones([2, 3], np.float32)
    ms_net = ShapeAdd()
    torch_net = ShapeAddT()
    ms_out = ms_net(Tensor(input_x), Tensor(input_y))
    torch_out = torch_net(tensor(input_x), tensor(input_y))
    ms_out_np = np.array(ms_out)
    torch_out_np = np.array(torch_out)
    compare_nparray(torch_out_np, ms_out_np, 1e-4, 1e-4)
    print("RUNTIME_COMPILE", ms_out_np, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", ms_out_np.shape, "RUNTIME_CACHE")
