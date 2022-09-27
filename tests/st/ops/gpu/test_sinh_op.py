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

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.ops import composite as C
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class NetSinh(nn.Cell):
    def __init__(self):
        super(NetSinh, self).__init__()
        self.sinh = P.Sinh()

    def construct(self, x):
        return self.sinh(x)


class SinhGradNet(nn.Cell):
    def __init__(self, network):
        super(SinhGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x, grad_np):
        grad_out = self.grad(self.network)(x, grad_np)
        return grad_out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sinh_fp16():
    """
    Feature: Sinh
    Description: test cases for Sinh of float16
    Expectation: the results are as expected
    """
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    input_x = Tensor(x_np)
    net = NetSinh()
    output_ms = net(input_x)
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = SinhGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.521, 1.176, 10.016, 12.234]).astype(np.float16)
    expect_grad_output = np.array([0.5635, 1.543, 30.19, 39.28]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sinh_fp32():
    """
    Feature: Sinh
    Description: test cases for Sinh of float32
    Expectation: the results are as expected
    """
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    input_x = Tensor(x_np)
    net = NetSinh()
    output_ms = net(input_x)
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = SinhGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([1.1752012, 3.6268604, 10.017875, 27.289917]).astype(np.float32)
    expect_grad_output = np.array([1.5430806, 7.5243917, 30.202988, 109.232925]).astype(np.float32)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sinh_fp64():
    """
    Feature: Sinh
    Description: test cases for Sinh of float64
    Expectation: the results are as expected
    """
    x_np = np.array([0.2, 0.9, 2.4, 8.8]).astype(np.float64)
    input_x = Tensor(x_np)
    net = NetSinh()
    output_ms = net(input_x)
    grad_np = np.array([0.2, 0.9, 2.4, 8.8]).astype(np.float64)
    grad_net = SinhGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([2.01336003e-01, 1.02651673e+00, 5.46622921e+00, 3.31712193e+03]).astype(np.float64)
    expect_grad_output = np.array([2.04013351e-01, 1.28977775e+00, 1.33366732e+01, 2.91906743e+04]).astype(np.float64)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)
