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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Roll(nn.Cell):

    def __init__(self, shift, axis):
        super(Roll, self).__init__()
        self.shift = shift
        self.axis = axis
        self.roll = nn.Roll(self.shift, self.axis)

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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roll_1d():
    """
    Feature: RightShift gpu TEST.
    Description: 1d test case for RightShift
    Expectation: the result match to numpy
    """
    x_np = np.array([-1, -5, -3, -14, 64]).astype(np.int8)
    x_grad_np = np.array([-1, -5, -3, -14, 64]).astype(np.int8)
    except_output = np.array([-5, -3, -14, 64, -1]).astype(np.int8)
    shift = 4
    axis = 0
    x_ms = Tensor(x_np)
    net = Roll(shift, axis)
    grad_net = RollGrad(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(x_grad_np))
    except_grad_output = np.array([64, -1, -5, -3, -14]).astype(np.int8)
    output_ms = net(x_ms)

    assert np.allclose(except_output, output_ms.asnumpy())
    assert np.allclose(except_grad_output, output_grad_ms[0].asnumpy())
