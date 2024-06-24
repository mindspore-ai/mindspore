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
import mindspore.context as context
from mindspore.ops.operations.math_ops import Lgamma
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class LgammaNet(nn.Cell):

    def __init__(self):
        super(LgammaNet, self).__init__()
        self.lgamma = Lgamma()

    def construct(self, input_x):
        return self.lgamma(input_x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lgamma_graph_float16():
    """
    Feature: ALL To ALL
    Description: test cases for Lgamma
    Expectation: the result match to torch
    """
    net = LgammaNet()
    x_ms = np.array([1, 0.4273, 9, -3.12, 122.345]).astype(np.float16)
    z_ms = net(Tensor(x_ms))
    expect = np.array([0.000e+00, 7.295e-01, 1.060e+01, 2.075e-01, 4.645e+02]).astype(np.float64)
    assert np.allclose(z_ms.asnumpy(), expect.astype(np.float16), 0.001, 0.001)



@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lgamma_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for Lgamma
    Expectation: the result match to torch
    """
    net = LgammaNet()
    x_ms = np.array([1, 0.4273, 9, -3.12, 12246.345]).astype(np.float32)
    z_ms = net(Tensor(x_ms))
    expect = np.array([0.0000000e+00, 7.2935897e-01, 1.0604603e+01,
                       1.9955045e-01, 1.0302450e+05]).astype(np.float32)
    assert np.allclose(z_ms.asnumpy(), expect, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lgamma_graph_float64():
    """
    Feature: ALL To ALL
    Description: test cases for Lgamma
    Expectation: the result match to torch
    """
    net = LgammaNet()
    x_ms = np.array([1, 0.4273, 9, -3.12, 12246.345]).astype(np.float64)
    z_ms = net(Tensor(x_ms))
    expect = np.array([0.00000000e+00, 7.29359005e-01, 1.06046029e+01,
                       1.99549382e-01, 1.03024502e+05]).astype(np.float64)
    assert np.allclose(z_ms.asnumpy(), expect, 0.00001, 0.00001)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lgamma_invalid_input():
    """
    Feature: ALL To ALL
    Description: test cases for Lgamma
    Expectation: throw type error
    """
    net = LgammaNet()
    try:
        net("invalid input")
    except TypeError:
        pass
