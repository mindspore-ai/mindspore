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
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor, jit


class CheckNumericsNet(nn.Cell):

    def __init__(self):
        super(CheckNumericsNet, self).__init__()
        self.checknumerics = P.CheckNumerics()

    @jit
    def construct(self, input_x):
        return self.checknumerics(input_x)


def check_numerics(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x_np = np.array([[6, 3], [2, 5]], dtype=np.float32)
    input_x_ms = Tensor(input_x_np)
    check_numerics_net = CheckNumericsNet()
    check_numerics_output = check_numerics_net(input_x_ms)
    check_numerics_expect = np.array([[6, 3], [2, 5]], dtype=np.float32)
    assert np.allclose(check_numerics_output.asnumpy(), check_numerics_expect,
                       loss, loss)


def check_numerics_pynative(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x_np = np.array([[1, 5], [2, 4]], dtype=np.float64)
    input_x_ms = Tensor(input_x_np)
    check_numerics_net = CheckNumericsNet()
    check_numerics_output = check_numerics_net(input_x_ms)
    check_numerics_expect = np.array([[1, 5], [2, 4]], dtype=np.float64)
    print(check_numerics_output)
    print(check_numerics_expect)
    assert np.allclose(check_numerics_output.asnumpy(), check_numerics_expect,
                       loss, loss)


def dyn_case():
    net = CheckNumericsNet()

    x_dyn = Tensor(shape=[None, None], dtype=ms.float64)
    net.set_inputs(x_dyn)

    x = Tensor(
        np.array([[0.42987306, 0.02847828, 0.59385591, 0.7040952, 0.27390435],
                  [0.32904094, 0.63063352, 0.70752448, 0.24763578, 0.99662956],
                  [0.66478424, 0.70580542, 0.92749155, 0.72736302, 0.24973136],
                  [0.79918445, 0.68613469, 0.9526593, 0.12412648,
                   0.15175918]]).astype(np.float64))
    out = net(x)

    expect_shape = (4, 5)
    assert out.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_check_numerics_dyn():
    """
    Feature: test CheckNumerics ops in gpu.
    Description: Test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    dyn_case()


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
