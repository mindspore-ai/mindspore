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


class NetSigmoidGrad(nn.Cell):
    def __init__(self):
        super(NetSigmoidGrad, self).__init__()
        self.sigmoid_grad = G.SigmoidGrad()

    def construct(self, y, dy):
        return self.sigmoid_grad(y, dy)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Sigmoid()

    def construct(self, x):
        return self.ops(x)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    x = np.random.randn(2, 3, 3, 4).astype(np.float32)
    y_expect = 1 / (1 + np.exp(-x))
    net = Net()
    out = net(Tensor(x))
    diff = out.asnumpy() - y_expect
    err = np.ones(shape=y_expect.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == y_expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sigmoid_grad():
    y = Tensor(np.array([[[[-1, 1, 2],
                           [1, -1, 1],
                           [2, 1, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[-11, 2, 4],
                            [-1, 1, -1],
                            [-4, 4, -4]]]]).astype(np.float32))

    expect = np.array([[[[22, 0, -8],
                         [0, -2, 0],
                         [8, 0, 8]]]]).astype(np.float32)

    error = np.ones(shape=[1, 1, 3, 3]) * 1.0e-6

    sigmoid_grad = NetSigmoidGrad()
    output = sigmoid_grad(y, dy)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(abs(diff) < error)
