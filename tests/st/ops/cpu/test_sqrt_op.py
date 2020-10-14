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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class NetSqrtGrad(nn.Cell):
    def __init__(self):
        super(NetSqrtGrad, self).__init__()
        self.sqrt_grad = G.SqrtGrad()

    def construct(self, x, dx):
        return self.sqrt_grad(x, dx)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Sqrt()

    def construct(self, x):
        return self.ops(x)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    x = np.abs(np.random.randn(2, 3, 3, 4)).astype(np.float32)
    y_expect = np.sqrt(x)
    net = Net()
    out = net(Tensor(x))
    diff = out.asnumpy() - y_expect
    err = np.ones(shape=y_expect.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == y_expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sqrt_grad():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [5.9, 6.1, 6],
                           [10, 1, -1]]]]).astype(np.float32))
    dx = Tensor(np.array([[[[1, 1, 1],
                            [2, 2, 2],
                            [3, 3, 3]]]]).astype(np.float32))
    expect = np.array([[[[-0.5, 0.5, 0.05,],
                         [0.16949153, 0.16393442, 0.16666667,],
                         [0.15, 1.5, -1.5,]]]]).astype(np.float32)
    error = np.ones(shape=[3, 3]) * 1.0e-6

    sqrt_grad = NetSqrtGrad()
    output = sqrt_grad(x, dx)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)
