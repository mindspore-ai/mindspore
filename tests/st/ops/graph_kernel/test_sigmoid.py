# Copyright 2021 Huawei Technologies Co., Ltd
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


class NetSigmoid(nn.Cell):
    def __init__(self):
        super(NetSigmoid, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        return self.sigmoid(x)


class NetSigmoidGrad(nn.Cell):
    def __init__(self):
        super(NetSigmoidGrad, self).__init__()
        self.sigmoid_grad = G.SigmoidGrad()

    def construct(self, y, dy):
        return self.sigmoid_grad(y, dy)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sigmoid():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))

    error = np.ones(shape=[1, 1, 3, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    net = NetSigmoid()
    result_open_gk = net(x)

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=False, device_target="GPU")
    net_beta = NetSigmoid()
    result_close_gk = net_beta(x)
    diff = result_open_gk.asnumpy() - result_close_gk.asnumpy()
    assert np.all(abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sigmoid_grad():
    y = Tensor(np.array([[[[-1, 1, 2],
                           [1, -1, 1],
                           [2, 1, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[-11, 2, 4],
                            [-1, 1, -1],
                            [-4, 4, -4]]]]).astype(np.float32))

    error = np.ones(shape=[1, 1, 3, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    net = NetSigmoidGrad()
    result_open_gk = net(y, dy)

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=False, device_target="GPU")
    net_beta = NetSigmoidGrad()
    result_close_gk = net_beta(y, dy)
    diff = result_open_gk.asnumpy() - result_close_gk.asnumpy()
    assert np.all(abs(diff) < error)
