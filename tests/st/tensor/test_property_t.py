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


class Net(nn.Cell):
    def construct(self, x):
        return x.T


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_t(mode):
    """
    Feature: tensor.T
    Description: Verify the result of T
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[1, 2, 3], [2, 3, 4]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[1, 2], [2, 3], [3, 4]]
    assert np.allclose(output.asnumpy(), expect_output)
    x1 = Tensor([])
    output1 = net(x1)
    expect_output1 = np.array([])
    assert np.allclose(output1.asnumpy(), expect_output1)
    x2 = Tensor(1)
    output2 = net(x2)
    expect_output2 = np.array(1)
    assert np.allclose(output2.asnumpy(), expect_output2)
