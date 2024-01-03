# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_dtype_fp32_to_bool():
    """
    Feature: Create a Tensor by another Tensor fp32 -> bool.
    Description: Create a Tensor by another Tensor.
    Expectation: success but throw warning.
    """
    input_ = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_ = ms.Tensor(input_)
    t = ms.Tensor(input_, dtype=ms.bool_)
    assert isinstance(t, ms.Tensor)
    assert t.shape == (2, 3, 4, 5)
    assert t.dtype == ms.bool_


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_dtype_int8_to_bool():
    """
    Feature: Create a Tensor by another Tensor int8 -> bool.
    Description: Create a Tensor by another Tensor.
    Expectation: success but throw warning.
    """
    a = Tensor([[1, 1], [1, 1]], ms.int8)
    b = Tensor(a, ms.bool_)
    expected = np.array([[True, True], [True, True]], np.bool_)
    assert np.array_equal(b.asnumpy(), expected)
