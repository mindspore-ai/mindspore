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
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops.operations.math_ops import Cholesky
from mindspore.nn import Cell


class CholeskyNet(Cell):
    def __init__(self):
        super().__init__()
        self.cholesky = Cholesky()

    def construct(self, x):
        return self.cholesky(x)


class CholeskyGradNet(Cell):
    def __init__(self, network):
        super(CholeskyGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cholesky_fp32():
    """
    Feature: Cholesky
    Description: test input data of float32
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([[10, 22], [22, 50]]).astype(np.float32)
    net = CholeskyNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([[1, 0], [0, 1]]).astype(np.float32)
    grad_net = CholeskyGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([[3.1622777, 0], [6.9570107, 1.2649117]])
    expect_grad_output = np.array([[2.071291, -0.869626], [-0.869626, 0.39528453]])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cholesky_fp64():
    """
    Feature: Cholesky
    Description: test input data of float64
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([[12.56, 27.28], [27.28, 60.5]]).astype(np.float64)
    net = CholeskyNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([[1, 0], [0, 1]]).astype(np.float64)
    grad_net = CholeskyGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([[3.544009, 0.], [7.697497, 1.1173816]])
    expect_grad_output = np.array([[2.252033, -0.9719036], [-0.9719037, 0.44747472]])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)
