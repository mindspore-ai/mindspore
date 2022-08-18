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
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function


class CheckNumericsNet(nn.Cell):
    def __init__(self):
        super(CheckNumericsNet, self).__init__()
        self.checknumerics = P.CheckNumerics()

    @ms_function
    def construct(self, input_x):
        return self.checknumerics(input_x)


def check_numerics(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x_np = np.array([[6, 3], [2, 5]], dtype=np.float32)
    input_x_ms = Tensor(input_x_np)
    check_numerics_net = CheckNumericsNet()
    check_numerics_output = check_numerics_net(input_x_ms)
    check_numerics_expect = np.array([[6, 3], [2, 5]], dtype=np.float32)
    assert np.allclose(check_numerics_output.asnumpy(), check_numerics_expect, loss, loss)


def check_numerics_pynative(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x_np = np.array([[1, 5], [2, 4]], dtype=np.float64)
    input_x_ms = Tensor(input_x_np)
    check_numerics_net = CheckNumericsNet()
    check_numerics_output = check_numerics_net(input_x_ms)
    check_numerics_expect = np.array([[1, 5], [2, 4]], dtype=np.float64)
    print(check_numerics_output)
    print(check_numerics_expect)
    assert np.allclose(check_numerics_output.asnumpy(), check_numerics_expect, loss, loss)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_check_numerics_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for CheckNumerics
    Expectation: the result match to tensorflow
    """
    check_numerics(loss=1.0e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_check_numerics_pynative_float64():
    """
    Feature: ALL To ALL
    Description: test cases for CheckNumerics
    Expectation: the result match to tensorflow
    """
    check_numerics_pynative(loss=1.0e-5)
