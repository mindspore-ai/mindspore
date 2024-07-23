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
import mindspore.context as context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import composite as C

context.set_context(device_target="Ascend")


class NetGrad(nn.Cell):
    def __init__(self):
        super(NetGrad, self).__init__()
        self.grad = G.RsqrtGrad()

    def construct(self, out, dout):
        return self.grad(out, dout)


class NetGradGrad(nn.Cell):
    def __init__(self, forward_net):
        super(NetGradGrad, self).__init__()
        self.forward_net = forward_net
        self.grad_ops = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, y, grad, dout):
        backward_net = self.grad_ops(self.forward_net)
        return backward_net(y, grad, dout)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rsqrt_grad_grad_float16():
    """
    Feature: Rsqrt Grad Grad operation
    Description: test the grad of RsqrtGrad kernel, with 4 dim input.
    Expectation: the output is same with numpy
    """
    x = np.random.randint(1, 10, (2, 1, 3, 3)).astype(np.float16)
    y = 1 / np.sqrt(x)

    grad = -0.5 * np.power(x, -1.5)
    dout = np.ones_like(x).astype(np.float16)

    grad_net = NetGrad()
    grad_grad_net = NetGradGrad(grad_net)
    dgrad_ms, _ = grad_grad_net(Tensor(y), Tensor(grad), Tensor(dout))

    expected = 3 / 4 * np.power(x, -2.5)
    assert np.allclose(dgrad_ms.asnumpy(), expected, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rsqrt_grad_grad_float32():
    """
    Feature: Rsqrt Grad Grad operation
    Description: test the grad of RsqrtGrad kernel, with 1 dim input.
    Expectation: the output is same with numpy
    """
    x = np.array([9., 3., 2., 2., 2.]).astype(np.float32)
    y = 1 / np.sqrt(x)

    grad = -0.5 * np.power(x, -1.5)
    dout = np.array([1, 1, 1, 1, 1]).astype(np.float32)

    grad_net = NetGrad()
    grad_grad_net = NetGradGrad(grad_net)
    dgrad_ms, _ = grad_grad_net(Tensor(y), Tensor(grad), Tensor(dout))

    expected = 3 / 4 * np.power(x, -2.5)
    assert np.allclose(dgrad_ms.asnumpy(), expected, 1e-4, 1e-4)
