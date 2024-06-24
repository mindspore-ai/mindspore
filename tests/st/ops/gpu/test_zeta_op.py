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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.math_ops import Zeta


class NetZeta(nn.Cell):

    def __init__(self):
        super(NetZeta, self).__init__()
        self.zeta = Zeta()

    def construct(self, x, y):
        return self.zeta(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_zeta_1d_input_float32_output_float32():
    """
    Feature: Zeta gpu TEST.
    Description: 1d test case for Zeta
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_ms = Tensor(np.array([3, 3, 9]).astype(np.float32))
    q_ms = Tensor(np.array([4, 2, 9]).astype(np.float32))
    net = NetZeta()
    z_ms = net(x_ms, q_ms)
    expect = np.array([4.0019866e-02, 2.02056915e-01, 4.4048485e-09])

    assert np.allclose(z_ms.asnumpy(), expect.astype(np.float32), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_zeta_1d_input_float64_output_float64():
    """
    Feature: Zeta gpu TEST.
    Description: 1d test case for Zeta
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    x_ms = Tensor(np.array([13, 5.3, 6, 10]).astype(np.float64))
    q_ms = Tensor(np.array([4.4, 21.2, -4.7, -3.7]).astype(np.float64))
    net = NetZeta()
    z_ms = net(x_ms, q_ms)
    expect = np.array([4.6569e-09, 5.0921e-07, 1.3805e+03, 1.6939e+05])

    assert np.allclose(z_ms.asnumpy(), expect.astype(np.float64), 0.0001, 0.0001)
