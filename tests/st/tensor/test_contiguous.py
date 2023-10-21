# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x):
        return x.contiguous()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_contiguous(mode):
    """
    Feature: countiguous
    Description: Verify the result of x
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    y = ops.transpose(x, (1, 0))
    Net()(y)
    y[:, 1] = 1
    expect_output = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(x.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_contiguous_pynative():
    """
    Feature: countiguous
    Description: Verify the result of x
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    y = ops.transpose(x, (1, 0))
    y.contiguous()
    y[:, 1] = 1
    expect_output = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(x.asnumpy(), expect_output)
