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
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations.math_ops as P


class NetGcd(nn.Cell):
    def __init__(self):
        super(NetGcd, self).__init__()
        self.gcd = P.Gcd()

    def construct(self, x1, x2):
        return self.gcd(x1, x2)


class NetLcm(nn.Cell):
    def __init__(self):
        super(NetLcm, self).__init__()
        self.lcm = P.Lcm()

    def construct(self, x1, x2):
        return self.lcm(x1, x2)


context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcd_int32():
    """
    Feature: Gcd
    Description: test cases for Gcd of int32
    Expectation: the results are as expected
    """
    x1_np = np.array([5, 10, 15]).astype(np.int32)
    x2_np = np.array([3, 4, 5]).astype(np.int32)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetGcd()
    output_ms = net(input_x1, input_x2)
    expect_output = np.gcd(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcd_int64():
    """
    Feature: Gcd
    Description: test cases for Gcd of int64
    Expectation: the results are as expected
    """
    x1_np = np.array([5, 10, 15]).astype(np.int64)
    x2_np = np.array([3]).astype(np.int64)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetGcd()
    output_ms = net(input_x1, input_x2)
    expect_output = np.gcd(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lcm_int32():
    """
    Feature: Lcm
    Description: test cases for Lcm of int32
    Expectation: the results are as expected
    """
    x1_np = np.array([5, 10, 15]).astype(np.int32)
    x2_np = np.array([3, 4, 5]).astype(np.int32)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetLcm()
    output_ms = net(input_x1, input_x2)
    expect_output = np.lcm(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lcm_int64():
    """
    Feature: Lcm
    Description: test cases for Lcm of int64
    Expectation: the results are as expected
    """
    x1_np = np.array([5, 10, 15]).astype(np.int64)
    x2_np = np.array([5]).astype(np.int64)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetLcm()
    output_ms = net(input_x1, input_x2)
    expect_output = np.lcm(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcd_int64_and_int32():
    """
    Feature: Gcd
    Description: test cases for Gcd of different dtype
    Expectation: the results are as expected
    """
    x1_np = np.array([10, 15, 20]).astype(np.int64)
    x2_np = np.array([3, 4, 5]).astype(np.int32)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetGcd()
    output_ms = net(input_x1, input_x2)
    expect_output = np.gcd(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lcm_int32_and_int64():
    """
    Feature: Lcm
    Description: test cases for Lcm of different dtype
    Expectation: the results are as expected
    """
    x1_np = np.array([10, 15, 20]).astype(np.int32)
    x2_np = np.array([5]).astype(np.int64)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetLcm()
    output_ms = net(input_x1, input_x2)
    expect_output = np.lcm(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)
