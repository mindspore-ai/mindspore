# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor, dtype

@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_h2d_copy():
    """
    Feature: test_h2d_copy
    Description: test host to device copy.
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_np = np.random.randn(512, 3, 224, 224).astype(np.float32)
    output_np = input_np + input_np

    input_tensor = Tensor(input_np, dtype=dtype.float32)
    output_tensor = input_tensor + input_tensor
    assert np.allclose(output_tensor.asnumpy(), output_np)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_h2d_copy_with_from_numpy():
    """
    Feature: test_h2d_copy_with_from_numpy.
    Description: test host to device copy.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_np = np.random.randn(512, 3, 224, 224).astype(np.float32)
    output_np = input_np + input_np

    input_tensor = Tensor.from_numpy(input_np).astype(dtype.float32)
    output_tensor = input_tensor + input_tensor
    assert np.allclose(output_tensor.asnumpy(), output_np)
