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
def test_acos_fp32():
    """
    Feature: acos kernel
    Description: test acos float32
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.float32)
    output_ms = P.ACos()(Tensor(x_np))
    output_np = np.arccos(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acos_fp16():
    """
    Feature: acos kernel
    Description: test acos float16
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.float16)
    output_ms = P.ACos()(Tensor(x_np))
    output_np = np.arccos(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acos_fp64():
    """
    Feature: acos kernel
    Description: test acos float64
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.float64)
    output_ms = P.ACos()(Tensor(x_np))
    output_np = np.arccos(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acos_complex64():
    """
    Feature: acos kernel
    Description: test acos complex64
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.complex64)
    x_np = x_np + 2j*x_np
    output_ms = P.ACos()(Tensor(x_np))
    output_np = np.arccos(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acos_complex128():
    """
    Feature: acos kernel
    Description: test acos complex128
    Expectation: just test
    """
    x_np = np.array([0.74, 0.04, 0.30, 0.56]).astype(np.complex128)
    x_np = x_np + 5j*x_np
    output_ms = P.ACos()(Tensor(x_np))
    output_np = np.arccos(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
