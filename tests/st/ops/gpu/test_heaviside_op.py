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
import mindspore.ops.operations.math_ops as P
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class NetHeaviside(nn.Cell):
    def __init__(self):
        super(NetHeaviside, self).__init__()
        self.heaviside = P.Heaviside()

    def construct(self, x1, x2):
        return self.heaviside(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_heaviside_fp16():
    """
    Feature: Heaviside
    Description: test cases for Heaviside of float16
    Expectation: the results are as expected
    """
    x1_np = np.array([-1.3, 0.000, 0.000, 1.234]).astype(np.float16)
    x2_np = np.array([0, -1.234, -1.234, 1]).astype(np.float16)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetHeaviside()
    output_ms = net(input_x1, input_x2)
    expect_output = np.array([0.0000, -1.2344, -1.2344, 1.0000]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_heaviside_fp32():
    """
    Feature: Heaviside
    Description: test cases for Heaviside of float32
    Expectation: the results are as expected
    """
    x1_np = np.array([-1, 0, 0, 1]).astype(np.float32)
    x2_np = np.array([0, -1, -1, 1]).astype(np.float32)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetHeaviside()
    output_ms = net(input_x1, input_x2)
    expect_output = np.array([0., -1., -1., 1.]).astype(np.float32)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_heaviside_fp64():
    """
    Feature: Heaviside
    Description: test cases for Heaviside of float64
    Expectation: the results are as expected
    """
    x1_np = np.array([-1.5, 0, 2.0]).astype(np.float64)
    x2_np = np.array([1.7]).astype(np.float64)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetHeaviside()
    output_ms = net(input_x1, input_x2)
    expect_output = np.array([0.0000, 1.7000, 1.0000]).astype(np.float64)
    assert np.allclose(output_ms.asnumpy(), expect_output)
