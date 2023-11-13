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


class Net(nn.Cell):
    def construct(self, x, indices, dim=None):
        return x.take_along_dim(indices, dim)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_take_along_dim(mode):
    """
    Feature: take_along_dim
    Description: Verify the result of take_along_dim when all the inputs are valid
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.array([[10, 30, 20], [60, 40, 50]]))
    indices = ms.Tensor([[0, 2, 1], [1, 2, 0]])
    output = net(x, indices, 1)
    expect_output = np.array([[10, 20, 30], [40, 50, 60]])
    assert np.allclose(output.asnumpy(), expect_output)
