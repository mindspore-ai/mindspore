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
import mindspore.ops.operations._grad_ops as G


class ReluNet(nn.Cell):
    def __init__(self):
        super(ReluNet, self).__init__()
        self.relu = P.ReLU()
        self.relu_grad = G.ReluGrad()

    def construct(self, x, dy):
        y = self.relu(x)
        dx = self.relu_grad(dy, y)
        return y, dx

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ReluV2():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=True)

    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[1, 0, 3],
                            [0, 1, 0],
                            [2, 1, 1]]]]).astype(np.float32))
    expect_y = np.array([[[[0, 1, 10,],
                           [1, 0, 1,],
                           [10, 1, 0.]]]]).astype(np.float32)
    expect_dx = np.array([[[[0, 0, 3],
                            [0, 0, 0],
                            [2, 1, 0]]]]).astype(np.float32)
    net = ReluNet()
    y, dx = net(Tensor(x), Tensor(dy))

    assert np.allclose(y.asnumpy(), expect_y)
    assert np.allclose(dx.asnumpy(), expect_dx)


class AddReluNet(nn.Cell):
    def __init__(self):
        super(AddReluNet, self).__init__()
        self.add = P.Add()
        self.relu = P.ReLU()
        self.relu_grad = G.ReluGrad()

    def construct(self, x1, x2, dy):
        y = self.add(x1, x2)
        y = self.relu(y)
        dx = self.relu_grad(dy, y)
        return y, dx


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_AddRelu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=True)

    x1 = Tensor(np.array([[[[-1, 1, 10],
                            [1, -1, 1],
                            [10, 1, -1]]]]).astype(np.float32))
    x2 = Tensor(np.array([[[[-1, 1, 10],
                            [1, -1, 1],
                            [10, 1, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[1, 0, 3],
                            [0, 1, 0],
                            [2, 1, 1]]]]).astype(np.float32))
    expect_y = np.array([[[[0, 2, 20],
                           [2, 0, 2],
                           [20, 2, 0]]]]).astype(np.float32)
    expect_dx = np.array([[[[0, 0, 3],
                            [0, 0, 0],
                            [2, 1, 0]]]]).astype(np.float32)
    net = AddReluNet()
    y, dx1 = net(Tensor(x1), Tensor(x2), Tensor(dy))

    assert np.allclose(y.asnumpy(), expect_y)
    assert np.allclose(dx1.asnumpy(), expect_dx)

class AddReluGradNet(nn.Cell):
    def __init__(self):
        super(AddReluGradNet, self).__init__()
        self.add = P.Add()
        self.relu = P.ReLU()
        self.relu_grad = G.ReluGrad()

    def construct(self, x, dy1, dy2):
        y = self.relu(x)
        dy = self.add(dy1, dy2)
        dx = self.relu_grad(dy, y)
        return y, dx


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_AddReluGrad():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=True)

    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    dy1 = Tensor(np.array([[[[1, 0, 3],
                             [0, 1, 0],
                             [2, 1, 1]]]]).astype(np.float32))
    dy2 = Tensor(np.array([[[[1, 0, 3],
                             [0, 1, 0],
                             [2, 1, 1]]]]).astype(np.float32))
    expect_y = np.array([[[[0, 1, 10,],
                           [1, 0, 1,],
                           [10, 1, 0.]]]]).astype(np.float32)
    expect_dx = np.array([[[[0, 0, 6],
                            [0, 0, 0],
                            [4, 2, 0]]]]).astype(np.float32)
    net = AddReluGradNet()
    y, dx1 = net(Tensor(x), Tensor(dy1), Tensor(dy2))

    assert np.allclose(y.asnumpy(), expect_y)
    assert np.allclose(dx1.asnumpy(), expect_dx)
