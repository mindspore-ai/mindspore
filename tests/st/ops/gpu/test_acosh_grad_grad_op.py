# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class NetAcoshGrad(nn.Cell):
    def __init__(self):
        super(NetAcoshGrad, self).__init__()
        self.acosh_grad = G.AcoshGrad()

    def construct(self, y, grad):
        return self.acosh_grad(y, grad)


class NetAcoshGradGrad(nn.Cell):
    def __init__(self, forward_net):
        super(NetAcoshGradGrad, self).__init__()
        self.forward_net = forward_net
        self.gradOps = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, y, grad, dout):
        backward_net = self.gradOps(self.forward_net)
        return backward_net(y, grad, dout)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def acosh_grad_grad_base(dtype, loss):
    np.random.seed(1)
    shape = (4, 2)
    y_np = (np.random.rand(*shape) * 10).astype(dtype)
    grad_np = (np.random.rand(*shape) * 20 - 10).astype(dtype)
    dout_np = (np.random.rand(*shape) * 20 - 10).astype(dtype)

    out_np = grad_np / np.sinh(y_np)
    dy_np = dout_np * out_np * (-1.0) / np.tanh(y_np)
    dgrad_np = dout_np / np.sinh(y_np)

    y_ms = Tensor(y_np)
    grad_ms = Tensor(grad_np)
    dout_ms = Tensor(dout_np)
    forward_net = NetAcoshGrad()
    net = NetAcoshGradGrad(forward_net)
    dy_ms, dgrad_ms = net(y_ms, grad_ms, dout_ms)

    assert np.allclose(dy_ms.asnumpy(), dy_np, loss, loss)
    assert np.allclose(dgrad_ms.asnumpy(), dgrad_np, loss, loss)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acosh_grad_grad_float16():
    acosh_grad_grad_base(np.float16, 2e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acosh_grad_grad_float32():
    acosh_grad_grad_base(np.float32, 1e-4)
