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
import mindspore.ops.operations._grad_ops as P
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acosgrad_fp32():
    """
    Feature: acos grad kernel
    Description: test acosgrad float32
    Expectation: just test
    """
    error = np.ones(4) * 1.0e-7
    x_np = np.array([0, -0.25, 0.5, 0.3]).astype(np.float32)
    dout_np = np.array([1, 1, 1, 1]).astype(np.float32)
    output_ms = P.ACosGrad()(Tensor(x_np), Tensor(dout_np))
    expect = np.array([-1, -1.0327955, -1.1547005, -1.0482849])
    diff = output_ms.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acosgrad_fp16():
    """
    Feature: acos grad kernel
    Description: test acosgrad float16
    Expectation: just test
    """
    error = np.ones(4) * 1.0e-3
    x_np = np.array([0, -0.25, 0.5, 0.3]).astype(np.float16)
    dout_np = np.array([1, 1, 1, 1]).astype(np.float16)
    output_ms = P.ACosGrad()(Tensor(x_np), Tensor(dout_np))
    expect = np.array([-1, -1.033, -1.154, -1.048])
    diff = output_ms.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acosgrad_fp64():
    """
    Feature: acos grad kernel
    Description: test acosgrad float64
    Expectation: just test
    """
    error = np.ones(4) * 1.0e-9
    x_np = np.array([0, -0.25, 0.5, 0.3]).astype(np.float64)
    dout_np = np.array([1, 1, 1, 1]).astype(np.float64)
    output_ms = P.ACosGrad()(Tensor(x_np), Tensor(dout_np))
    expect = -dout_np / np.sqrt(1 - x_np * x_np)
    diff = output_ms.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acosgrad_complex64():
    """
    Feature: acos grad kernel
    Description: test acosgrad complex64
    Expectation: just test
    """
    x_np = np.array([0, -0.25, 0.5, 0.3]).astype(np.complex64)
    x_np = x_np - 2j*x_np
    dout_np = np.array([1, 1, 1, 1]).astype(np.complex64)
    output_ms = P.ACosGrad()(Tensor(x_np), Tensor(dout_np))
    expect = -dout_np / np.conjugate(np.sqrt(1 - x_np * x_np))
    assert np.allclose(output_ms.asnumpy(), expect.astype(np.complex64), 1e-5, 1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_acosgrad_complex128():
    """
    Feature: acos grad kernel
    Description: test acosgrad complex128
    Expectation: just test
    """
    x_np = np.array([0, -0.25, 0.5, 0.3]).astype(np.complex128)
    x_np = x_np + 5j*x_np
    dout_np = np.array([1, 1, 1, 1]).astype(np.complex128)
    output_ms = P.ACosGrad()(Tensor(x_np), Tensor(dout_np))
    expect = -dout_np / np.conjugate(np.sqrt(1 - x_np * x_np))
    np.allclose(output_ms.asnumpy(), expect.astype(np.complex128), 1e-6, 1e-6)
