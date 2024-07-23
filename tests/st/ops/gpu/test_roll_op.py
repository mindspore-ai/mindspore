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
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Roll(nn.Cell):

    def __init__(self, shift, axis):
        super(Roll, self).__init__()
        self.shift = shift
        self.axis = axis
        self.roll = P.Roll(self.shift, self.axis)

    def construct(self, x):
        return self.roll(x)


class RollGrad(nn.Cell):
    def __init__(self, network):
        super(RollGrad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_roll_1d():
    """
    Feature: RightShift gpu TEST.
    Description: 1d test case for RightShift
    Expectation: the result match to numpy
    """
    x_np = np.array([-1, -5, -3, -14, 64]).astype(np.int8)
    x_grad_np = np.array([-1, -5, -3, -14, 64]).astype(np.int8)
    shift = 4
    axis = 0
    net = Roll(shift, axis)
    output_ms = net(Tensor(x_np))
    except_output = np.array([-5, -3, -14, 64, -1]).astype(np.int8)

    grad_net = RollGrad(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(x_grad_np))
    except_grad_output = np.array([64, -1, -5, -3, -14]).astype(np.int8)

    assert np.allclose(except_output, output_ms.asnumpy())
    assert np.allclose(except_grad_output, output_grad_ms[0].asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_roll_exception_1():
    """
    Feature: exception gpu TEST.
    Description: Test the case that shift has different size with axis
    Expectation: throw error info
    """
    x_np = np.arange(5).astype(np.float32)
    shift = 2
    axis = (0, -1, 0)
    try:
        _ = ms.ops.roll(Tensor(x_np), shift, dims=axis)
    except ValueError:
        assert True


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_roll_exception_2():
    """
    Feature: exception gpu TEST.
    Description: exception case when shifts is empty
    Expectation: throw error info
    """
    input_x = Tensor(np.random.uniform(-10, 10, size=[5, 5])).astype(ms.float32)
    shifts = ()
    axis = 0
    try:
        _ = ms.ops.roll(input_x, shifts, dims=axis)
    except ValueError:
        assert True
