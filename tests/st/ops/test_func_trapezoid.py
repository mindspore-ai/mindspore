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
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, y, x):
        output = ops.trapezoid(y, x)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_trapezoid_normal(mode):
    """
    Feature: trapezoid
    Description: Verify the result of trapezoid
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    y = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]).astype(np.float32))
    x = ms.Tensor(np.array([[1, 2, 3], [1, 3, 5], [1, 4, 7]]).astype(np.float32))
    out = net(y, x)
    expect_out = np.array([2., 4., 6.], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)
