# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Dynamic shape operators network definition for GPTO test"""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, context, ops
import mindspore.communication as comm

class GPTODynNet(nn.Cell):
    """Class for dynamic shape operators network"""
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        out = self.matmul(x, y)
        out = self.relu(out)
        return out

def test_gpto_net_dynamic_ops():
    """
    Feature: this function test the GPTO module in KBK
    Description: the input is a net with comp operators with dynamic shapes
    Expectation: the test should pass without any error and exception
    """
    context.set_context(mode=context.GRAPH_MODE, jit_level="O0")
    comm.init()
    net = GPTODynNet()

    x_dyn = Tensor(shape=[None, 4], dtype=ms.float32)
    y_dyn = Tensor(shape=[4, 3], dtype=ms.float32)
    net.set_inputs(x_dyn, y_dyn)

    a = Tensor(np.random.randn(2, 4).astype(np.float32))
    b = Tensor(np.random.randn(4, 3).astype(np.float32))
    net(a, b)


test_gpto_net_dynamic_ops()
