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
from mindspore.ops import composite as C
from mindspore.ops.operations import _grad_ops as G

context.set_context(device_target="Ascend")


class NetGrad(nn.Cell):
    def __init__(self, data_format):
        super(NetGrad, self).__init__()
        self.grad = G.BiasAddGrad(data_format)

    def construct(self, dout):
        return self.grad(dout)


class NetGradGrad(nn.Cell):
    def __init__(self, forward_net):
        super(NetGradGrad, self).__init__()
        self.forward_net = forward_net
        self.grad_ops = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, dy, dout):
        backward_net = self.grad_ops(self.forward_net)
        return backward_net(dy, dout)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_biasadd_high_grad_dim2_float32():
    """
    Feature: Biasadd Grad Grad operation
    Description: test the high grad of Rsqrt. Input tensor has 2 dims, float32 type.
    Expectation: the output is same with tensorflow
    """
    x = np.arange(1, 7).reshape((2, 3)).astype(np.float32)
    b = np.ones(shape=(3,)).astype(np.float32)

    dy = Tensor(np.ones_like(x).astype(np.float32))
    dout = Tensor(np.ones_like(b).astype(np.float32))

    grad_net = NetGrad("NCHW")
    grad_grad_net = NetGradGrad(grad_net)
    dgrad_ms = grad_grad_net(dy, dout)

    expected = np.ones_like(x)
    assert np.allclose(dgrad_ms[0].asnumpy(), expected, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_biasadd_high_grad_dim4_float16():
    """
    Feature: Biasadd Grad Grad operation
    Description: test the high grad of Rsqrt. Input tensor has 4 dims, float16 type.
    Expectation: the output is same with tensorflow
    """
    x = np.random.randn(3, 2, 3, 3).astype(np.float16)
    b = np.random.randn(2).astype(np.float16)

    dy = Tensor(np.ones_like(x).astype(np.float16))
    dout = Tensor(np.ones_like(b).astype(np.float16))

    grad_net = NetGrad("NCHW")
    grad_grad_net = NetGradGrad(grad_net)
    dgrad_ms = grad_grad_net(dy, dout)

    expected = np.ones_like(x)
    assert np.allclose(dgrad_ms[0].asnumpy(), expected, 1e-4, 1e-4)
