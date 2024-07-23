# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import composite as C

context.set_context(device_target="Ascend")

class NetSoftplusGrad(nn.Cell):
    def __init__(self):
        super(NetSoftplusGrad, self).__init__()
        self.softplusGrad = G.SoftplusGrad()

    def construct(self, grad, x):
        return self.softplusGrad(grad, x)

class NetSoftplusGradGrad(nn.Cell):
    def __init__(self, forward_net):
        super(NetSoftplusGradGrad, self).__init__()
        self.forward_net = forward_net
        self.gradOps = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, grad, x, dout):
        backward_net = self.gradOps(self.forward_net)
        return backward_net(grad, x, dout)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softplus_grad_grad(data_type, mode):
    """
    Feature: softplus_grad_grad
    Description: test cases for softplus_grad_grad
    Expectation: the result match to torch
    """
    dy = np.random.randn(2, 9, 7, 4, 8, 8).astype(np.float32)
    x = np.random.randn(2, 9, 7, 4, 8, 8).astype(np.float32)
    dout = np.random.randn(2, 9, 7, 4, 8, 8).astype(np.float32)

    dy_ms = Tensor(dy)
    x_ms = Tensor(x)
    dout_ms = Tensor(dout)

    softplus_grad = NetSoftplusGrad()
    grad_grad_net = NetSoftplusGradGrad(softplus_grad)
    _, d2x_ms = grad_grad_net(dy_ms, x_ms, dout_ms)

    d2x_np = dy * dout / (np.exp(-x) + 2.0 + np.exp(x))

    np.allclose(d2x_ms.asnumpy(), d2x_np, 1e-4, 1e-4)
