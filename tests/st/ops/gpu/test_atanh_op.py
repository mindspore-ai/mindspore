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
from mindspore.ops import operations as P
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


def atanh(x):
    return 0.5 * np.log((1. + x) / (1. - x))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_fp64():
    """
    Feature: Gpu Atanh kernel.
    Description: Double dtype input.
    Expectation: success.
    """
    x_np = np.array([[-0.16595599, 0.44064897, -0.99977124, -0.39533487],
                     [-0.7064882, -0.8153228, -0.62747955, -0.30887854],
                     [-0.20646505, 0.07763347, -0.16161098, 0.370439]]).astype(np.float64)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_fp32():
    """
    Feature: Gpu Atanh kernel.
    Description: Float32 dtype input.
    Expectation: success.
    """
    x_np = np.array([[-0.16595599, 0.44064897, -0.99977124, -0.39533487],
                     [-0.7064882, -0.8153228, -0.62747955, -0.30887854],
                     [-0.20646505, 0.07763347, -0.16161098, 0.370439]]).astype(np.float32)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_fp16():
    """
    Feature: Gpu Atanh kernel.
    Description: Float16 dtype input.
    Expectation: success.
    """
    x_np = np.array([[-0.16595599, 0.44064897, -0.99977124, -0.39533487],
                     [-0.7064882, -0.8153228, -0.62747955, -0.30887854],
                     [-0.20646505, 0.07763347, -0.16161098, 0.370439]]).astype(np.float16)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect, 1e-3, 1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_complex64():
    """
    Feature: Gpu Atanh kernel.
    Description: Complex64 dtype input.
    Expectation: success.
    """
    x_np = np.array([[2+3j, 4+5j, 6-7j, 8+9j],
                     [1+3j, 2+5j, 5-7j, 7+9j],
                     [3+3j, 4+5j, 4-7j, 6+9j]]).astype(np.complex64)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_complex128():
    """
    Feature: Gpu Atanh kernel.
    Description: Complex128 dtype input.
    Expectation: success.
    """
    x_np = np.array([[2+3j, 4+5j, 6-7j, 8+9j],
                     [1+3j, 2+5j, 5-7j, 7+9j],
                     [3+3j, 4+5j, 4-7j, 6+9j]]).astype(np.complex128)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect)


def test_atanh_forward_tensor_api(nptype):
    """
    Feature: test atanh forward tensor api for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([0, -0.5]).astype(nptype))
    output = x.atanh()
    expected = np.array([0.0, -0.54930615]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_forward_float32_tensor_api():
    """
    Feature: test atanh forward tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_atanh_forward_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_atanh_forward_tensor_api(np.float32)
