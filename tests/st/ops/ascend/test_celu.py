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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class TestCeluNet(nn.Cell):
    def __init__(self, alpha):
        super(TestCeluNet, self).__init__()
        self.celu = P.CeLU(alpha)

    def construct(self, x):
        return self.celu(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_celu_forward_float32():
    """
    Feature: test celu forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    celu_net = TestCeluNet(1.)
    x = Tensor(np.array([-2.0, -1.0, 1.0, 2.0]).astype(np.float32))
    output = celu_net(x)

    expect = np.array([-0.86468184, -0.6321212, 1., 2.]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_celu_dynamic_updates():
    """
    Feature: test celu dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    celu_net = TestCeluNet(1.)
    x = Tensor(np.array([[-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0]]).astype(np.float32))
    x_dy = Tensor(shape=(2, None), dtype=mindspore.float32)
    celu_net.set_inputs(x_dy)
    output = celu_net(x)

    expect = np.array([[-0.86468184, -0.6321212, 1., 2.], [-0.86468184, -0.6321212, 1., 2.]]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
