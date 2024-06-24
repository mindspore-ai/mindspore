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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import composite as C
from mindspore.ops import operations as P


class SoftplusNet(nn.Cell):
    def __init__(self):
        super(SoftplusNet, self).__init__()
        self.softplus = P.Softplus()

    def construct(self, x):
        return self.softplus(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True)
        self.network = network

    def construct(self, input_data):
        gout = self.grad(self.network)(input_data)
        return gout


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_dynamic_shape_softplus_grad(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for SoftplusGrad dynamic shape.
    Expectation: the result match to numpy
    """
    np.random.seed(0)
    x_np = np.random.randn(2, 3, 4).astype(dtype)
    expect = np.exp(x_np) / (1 + np.exp(x_np))
    loss = 1e-3
    net = SoftplusNet()
    grad_net = Grad(net)
    x_tensor = Tensor(x_np)
    dy_shape = [None for _ in x_tensor.shape]
    x_dyn = Tensor(shape=dy_shape, dtype=x_tensor.dtype)
    grad_net.set_inputs(x_dyn)

    # Graph mode
    context.set_context(mode=context.GRAPH_MODE)
    ms_result = grad_net(x_tensor)[0]
    np.testing.assert_allclose(expect, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = grad_net(x_tensor)[0]
    np.testing.assert_allclose(expect, ms_result.asnumpy(), rtol=loss, atol=loss)
