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
np.random.seed(1)


def atanh(x):
    return 0.5 * np.log((1. + x) / (1. - x))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_fp64():
    """
    Feature: Gpu Atanh kernel.
    Description: Double dtype input.
    Expectation: success.
    """
    x_np = np.random.uniform(-1, 1, size=(3, 4)).astype(np.float64)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_fp32():
    """
    Feature: Gpu Atanh kernel.
    Description: Float32 dtype input.
    Expectation: success.
    """
    x_np = np.random.uniform(-1, 1, size=(3, 4)).astype(np.float32)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atanh_fp16():
    """
    Feature: Gpu Atanh kernel.
    Description: Float16 dtype input.
    Expectation: success.
    """
    x_np = np.random.uniform(-1, 1, size=(3, 4)).astype(np.float16)
    output_ms = P.Atanh()(Tensor(x_np))
    expect = atanh(x_np)
    assert np.allclose(output_ms.asnumpy(), expect, 1e-3, 1e-3)
