# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Test high order grad with respect to parameter first, then input."""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore import ParameterTuple, Parameter
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()
        weight_np = np.array([2, 2]).astype(np.float32)
        self.weight = Parameter(Tensor(weight_np), name="weight", requires_grad=True)

    def construct(self, x):
        x_square = self.mul(x, x)
        x_square_z = self.mul(x_square, self.weight)
        output = self.mul(x_square_z, self.weight)
        return output


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, x):
        output = self.grad(self.network, self.params)(x)
        return output


class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, x):
        output = self.grad(self.network)(x)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_sit_high_order_grad_params():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([1, 1]).astype(np.float32))
    net = Net()
    first_grad = Grad(net)
    second_grad = GradSec(first_grad)
    grad = second_grad(x)
    assert (grad[0].asnumpy() == np.array([8, 8])).all()
