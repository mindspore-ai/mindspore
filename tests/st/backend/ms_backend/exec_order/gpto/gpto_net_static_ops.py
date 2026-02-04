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
"""Static shape operators network for GPTO test"""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, context, ops
import mindspore.communication as comm

class GPTONet(nn.Cell):
    """Class for static shape operators network"""
    def __init__(self):
        super().__init__()
        self.add_branch_1 = ops.Add()
        self.sub_branch_1 = ops.Sub()
        self.sub_branch_2 = ops.Sub()
        self.allreduce_branch_2 = ops.AllReduce()
        self.mul_output = ops.Mul()

    def overlap_net(self, x, y):
        o1 = self.add_branch_1(x, y)
        o1 = self.sub_branch_1(x, o1)
        o2 = self.sub_branch_2(x, y)
        o2 = self.allreduce_branch_2(o2)
        out = self.mul_output(o1, o2)
        return out

    def construct(self, x, y):
        out = self.overlap_net(x, y)
        return out

def test_gpto_net_static_ops():
    """
    Feature: this function test the GPTO module in KBK
    Description: the input is a net with comp and comm operators with static shapes, gpto will overlap them
    Expectation: the test should pass without any error and exception
    """
    context.set_context(mode=context.GRAPH_MODE, jit_level="O0")
    comm.init()
    net = GPTONet()
    x = Tensor(np.random.randn(512, 512), dtype=ms.float32)
    y = Tensor(np.random.randn(512, 512), dtype=ms.float32)
    net(x, y)

test_gpto_net_static_ops()
