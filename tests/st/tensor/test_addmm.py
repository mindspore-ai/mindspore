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
    def construct(self, x, y, z, beta, alpha):
        return x.addmm(y, z, beta=beta, alpha=alpha)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_addmm(mode):
    """
    Feature: tensor.addmm
    Description: Verify the result of addmm
    Expectation: success
    """
    ms.set_context(mode=mode)
    arr1 = np.arange(9).astype(np.float32).reshape((3, 3))
    arr2 = np.arange(12).astype(np.float32).reshape((3, 4))
    arr3 = np.arange(12).astype(np.float32).reshape((4, 3))
    x = Tensor(arr1)
    y = Tensor(arr2)
    z = Tensor(arr3)
    net = Net()
    output = net(x, y, z, 0.5, 2)
    expect_output = np.array([[84.0000, 96.5000, 109.0000],
                              [229.5000, 274.0000, 318.5000],
                              [375.0000, 451.5000, 528.0000]])
    assert np.allclose(output.asnumpy(), expect_output)
