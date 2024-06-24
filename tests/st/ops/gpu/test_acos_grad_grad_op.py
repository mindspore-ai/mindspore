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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.composite import GradOperation


class NetAcosGrad(nn.Cell):
    def __init__(self):
        super(NetAcosGrad, self).__init__()
        self.acos_grad = G.ACosGrad()

    @jit
    def construct(self, x, dy):
        return self.acos_grad(x, dy)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, x, grad, dout):
        return self.grad(self.network)(x, grad, dout)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("fp_type, error_magnitude, mode", [
    (np.float16, 1.0e-3, context.PYNATIVE_MODE),
    (np.float32, 1.0e-6, context.PYNATIVE_MODE),
    (np.float16, 1.0e-3, context.GRAPH_MODE),
    (np.float32, 1.0e-6, context.GRAPH_MODE)
])
def test_acos_grad_grad(fp_type, error_magnitude, mode):
    x = Tensor(np.array([0, -0.25, 0.5, 0.3]).astype(fp_type))
    grad = Tensor(np.array([0, -0.25, 0.5, 0.3]).astype(fp_type))
    dout = Tensor(np.array([2, 2, 2, 2]).astype(fp_type))

    expect_ddy = np.array([-2, -2.0655911, -2.3094011, -2.0965697]).astype(fp_type)
    expect_d2x = np.array([0, -0.1377061, -0.7698004, -0.2073530]).astype(fp_type)

    error = np.ones(4) * error_magnitude

    context.set_context(mode=mode, device_target="GPU")
    acos_grad_grad = Grad(NetAcosGrad())
    d2x, ddy = acos_grad_grad(x, grad, dout)
    diff0 = ddy.asnumpy() - expect_ddy
    diff1 = d2x.asnumpy() - expect_d2x
    assert np.all(abs(diff0) < error)
    assert np.all(abs(diff1) < error)
