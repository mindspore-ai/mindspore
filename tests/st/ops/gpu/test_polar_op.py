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
import mindspore.ops.operations.math_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function


class PolarNet(nn.Cell):
    def __init__(self):
        super(PolarNet, self).__init__()
        self.polar = P.Polar()

    @ms_function
    def construct(self, ms_abs, ms_angle):
        return self.polar(ms_abs, ms_angle)


def polar(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    np_abs = np.array([1, 2, 3, 4]).astype(np.float32)
    np_angle = np.array([np.pi/2, 5*np.pi/4, 3*np.pi/2, 2*np.pi/3]).astype(np.float32)
    ms_abs = Tensor(np_abs)
    ms_angle = Tensor(np_angle)
    net = PolarNet()
    output = net(ms_abs, ms_angle)
    expected = [-4.3711388e-08+1.j, -1.4142137e+00-1.4142134j, 3.5774640e-08-3.j, -2.0000002e+00+3.4641016j]
    assert np.allclose(output.asnumpy(), expected, loss, loss)


def polar_pynative(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    np_abs = np.array([1, 2, 3, 4]).astype(np.float64)
    np_angle = np.array([np.pi/2, 5*np.pi/4, 3*np.pi/2, 2*np.pi/3]).astype(np.float64)
    ms_abs = Tensor(np_abs)
    ms_angle = Tensor(np_angle)
    net = PolarNet()
    output = net(ms_abs, ms_angle)
    expected = [6.12323400e-17+1.j, -1.41421356e+00-1.41421356j, -5.51091060e-16-3.j, -2.00000000e+00+3.46410162j]
    assert np.allclose(output.asnumpy(), expected, loss, loss)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_polar_graph_float():
    """
    Feature: ALL To ALL
    Description: test cases for Polar
    Expectation: the result match to pytorch
    """
    polar(loss=1.0e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_polar_pynative_double():
    """
    Feature: ALL To ALL
    Description: test cases for Polar
    Expectation: the result match to pytorch
    """
    polar_pynative(loss=1.0e-5)
    