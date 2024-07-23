# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
context.set_context(save_graphs=True, save_graphs_path='./SoftmaxGrad_ir/')


class OriginNet(nn.Cell):
    def __init__(self):
        super(OriginNet, self).__init__()
        self.mul = P.Mul()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.sub = P.Sub()

    def construct(self, x, y):
        mul = self.mul(x, y)
        reduce_sum = self.reduce_sum(mul, -1)
        res = self.sub(y, reduce_sum)
        res = self.mul(x, res)
        return res


def numpy_func(x, y):
    mul = x * y
    reduce_sum = np.sum(mul, -1, keepdims=True)
    res = y - reduce_sum
    res = x * res
    return res


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softmax_grad_fusion():
    """
    Feature: SoftmaxGrad Fusion test
    Description: The output is correct after fusion
    Expectation: success
    """
    x_np = np.random.rand(9721, 21)
    y_np = np.random.rand(9721, 21)
    x = Tensor(x_np, ms.float32)
    y = Tensor(y_np, ms.float32)
    net = OriginNet()
    output = net(x, y)
    expect = numpy_func(x_np, y_np)
    assert np.allclose(output.asnumpy(), expect)
