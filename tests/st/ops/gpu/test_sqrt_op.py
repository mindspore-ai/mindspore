# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore.ops import operations as P


class SqrtNet(nn.Cell):
    def __init__(self):
        super(SqrtNet, self).__init__()
        self.ops = P.Sqrt()

    def construct(self, x):
        return self.ops(x)


def sqrt(nptype):
    """
    Feature: ALL To ALL
    Description: a function to test sqrt accuracy.
    Expectation: the result match to numpy
    """
    np.random.seed(0)
    x_np = np.random.rand(2, 3, 4, 4).astype(nptype)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_ms = P.Sqrt()(Tensor(x_np))
    output_np = np.sqrt(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sqrt_float16():
    sqrt(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sqrt_float32():
    sqrt(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sqrt_float64():
    sqrt(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rsqrt():
    np.random.seed(0)
    x_np = np.random.rand(2, 3, 4, 4).astype(np.float32)

    output_ms = P.Rsqrt()(Tensor(x_np))
    output_np = 1 / np.sqrt(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_sqrt_dy_shape(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Sqrt dynamic shape
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x_np = np.abs(np.random.randn(2, 3, 3, 4)).astype(dtype)
    benchmark_output = np.sqrt(input_x_np)
    loss = 1e-6
    sqrt_net = SqrtNet()
    real_input = Tensor(input_x_np)
    dy_shape = [None for _ in input_x_np.shape]
    input_dyn = Tensor(shape=dy_shape, dtype=real_input.dtype)
    sqrt_net.set_inputs(input_dyn)
    ms_result = sqrt_net(real_input)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = sqrt_net(real_input)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
