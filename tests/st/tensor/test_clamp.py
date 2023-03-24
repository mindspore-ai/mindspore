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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class ClampNet(nn.Cell):
    def construct(self, x, min_value, max_value):
        return x.clamp(min_value, max_value)


class ClipNet(nn.Cell):
    def construct(self, x, min_value, max_value):
        return x.clip(min_value, max_value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clamp(mode):
    """
    Feature: test Tensor.clamp
    Description: Verify the result of Tensor.clamp
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = ClampNet()
    output_ms_case_1 = net(x, 5, 20)
    expect_output_case_1 = np.clip(x_np, 5, 20)
    output_ms_case_2 = net(x, 20, 5)
    expect_output_case_2 = np.clip(x_np, 20, 5)
    assert np.allclose(output_ms_case_1.asnumpy(), expect_output_case_1)
    assert np.allclose(output_ms_case_2.asnumpy(), expect_output_case_2)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clip(mode):
    """
    Feature: test Tensor.clip
    Description: Verify the result of Tensor.clip
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = ClipNet()
    output_ms_case_1 = net(x, 5, 20)
    expect_output_case_1 = np.clip(x_np, 5, 20)
    output_ms_case_2 = net(x, 20, 5)
    expect_output_case_2 = np.clip(x_np, 20, 5)
    assert np.allclose(output_ms_case_1.asnumpy(), expect_output_case_1)
    assert np.allclose(output_ms_case_2.asnumpy(), expect_output_case_2)
