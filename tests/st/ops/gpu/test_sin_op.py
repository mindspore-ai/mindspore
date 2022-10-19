# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

def sin(nptype):
    if nptype == np.complex64 or np.complex128:
        x_np = np.random.rand(2, 3, 4, 4).astype(nptype) + 2j*np.random.rand(2, 3, 4, 4).astype(nptype)
    x_np = np.random.rand(2, 3, 4, 4).astype(nptype)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_ms = P.Sin()(Tensor(x_np))
    output_np = np.sin(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    output_ms = P.Sin()(Tensor(x_np))
    output_np = np.sin(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sin_float16():
    """
    Feature: sin kernel
    Description: test sin float16
    Expectation: just test
    """
    sin(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sin_float32():
    """
    Feature: sin kernel
    Description: test sin float32
    Expectation: just test
    """
    sin(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sin_float64():
    """
    Feature: sin kernel
    Description: test sin float64
    Expectation: just test
    """
    sin(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sin_complex64():
    """
    Feature: sin kernel
    Description: test sin complex64
    Expectation: just test
    """
    sin(np.complex64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sin_complex128():
    """
    Feature: sin kernel
    Description: test sin complex128
    Expectation: just test
    """
    sin(np.complex128)
