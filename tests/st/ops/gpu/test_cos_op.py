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

def cos(nptype):
    x_np = np.random.rand(2, 3, 4, 4).astype(nptype)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_ms = P.Cos()(Tensor(x_np))
    output_np = np.cos(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    output_ms = P.Cos()(Tensor(x_np))
    output_np = np.cos(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cos_float16():
    """
    Feature: cos kernel
    Description: test cos float16
    Expectation: just test
    """
    cos(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cos_float32():
    """
    Feature: cos kernel
    Description: test cos float32
    Expectation: just test
    """
    cos(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cos_float64():
    """
    Feature: cos kernel
    Description: test cos float64
    Expectation: just test
    """
    cos(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cos_complex64():
    """
    Feature: cos kernel
    Description: test cos complex64
    Expectation: just test
    """
    cos(np.complex64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cos_complex128():
    """
    Feature: cos kernel
    Description: test cos complex128
    Expectation: just test
    """
    cos(np.complex128)
