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

from mindspore import Tensor
from mindspore.ops.operations import math_ops as P


class NetAngle(nn.Cell):
    def __init__(self):
        super().__init__()
        self.angle = P.Angle()

    def construct(self, a):
        return self.angle(a)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_angle_pynative():
    """
    Feature: Angle
    Description: The input tensor. types: complex64, complex128
    Expectation: success: return a Tensor, has the float32 or float64 type and the same shape as input.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([-2.25 + 4.75j, 3.25 + 5.75j]).astype(np.complex64)
    net = NetAngle()
    output = net(Tensor(x_np))
    expect = np.angle(x_np)
    assert np.allclose(output.asnumpy(), expect, 1e-4, 1e-4)

    x_np = np.array([-2.25 + 4.75j, 3.25 + 5.75j]).astype(np.complex128)
    net = NetAngle()
    output = net(Tensor(x_np))
    expect = np.angle(x_np)
    assert np.allclose(output.asnumpy(), expect, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_angle_graph():
    """
    Feature: Angle
    Description: The input tensor. types: complex64, complex128
    Expectation: success: return a Tensor, has the float32 or float64 type and the same shape as input.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([-2.25 + 4.75j, 3.25 + 5.75j]).astype(np.complex64)
    net = NetAngle()
    output = net(Tensor(x_np))
    expect = np.angle(x_np)
    assert np.allclose(output.asnumpy(), expect, 1e-4, 1e-4)

    x_np = np.array([-2.25 + 4.75j, 3.25 + 5.75j]).astype(np.complex128)
    net = NetAngle()
    output = net(Tensor(x_np))
    expect = np.angle(x_np)
    assert np.allclose(output.asnumpy(), expect, 1e-5, 1e-5)
