# Copyright 2020 Huawei Technologies Co., Ltd
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

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asin_fp32():
    """
    Feature: asin kernel
    Description: test asin float32
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.float32)
    output_ms = P.Asin()(Tensor(x_np))
    output_np = np.arcsin(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asin_fp16():
    """
    Feature: asin kernel
    Description: test asin float16
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.float16)
    output_ms = P.Asin()(Tensor(x_np))
    output_np = np.arcsin(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asin_fp64():
    """
    Feature: asin kernel
    Description: test asin float64
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.float64)
    output_ms = P.Asin()(Tensor(x_np))
    output_np = np.arcsin(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asin_complex64():
    """
    Feature: asin kernel
    Description: test asin complex64
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.complex64)
    x_np = x_np + 2j*x_np
    output_ms = P.Asin()(Tensor(x_np))
    output_np = np.arcsin(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asin_complex128():
    """
    Feature: asin kernel
    Description: test asin complex128
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.complex128)
    x_np = x_np + 5j*x_np
    output_ms = P.Asin()(Tensor(x_np))
    output_np = np.arcsin(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


def test_asin_tensor_api(nptype):
    """
    Feature: test asin tensor api.
    Description: test inputs given their dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]).astype(nptype))
    output = x.asin()
    expected = np.array([0.8330704, 0.04001067, 0.30469266, 0.5943858]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_asin_float32_tensor_api():
    """
    Feature: test asin tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_asin_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_asin_tensor_api(np.float32)
