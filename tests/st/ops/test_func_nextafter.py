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
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, y):
        output = ops.nextafter(x, y)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nextafter(mode):
    """
    Feature: ops.nextafter
    Description: Verify the result of ops.nextafter
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.asarray([0.4]), ms.float32)
    y = Tensor(np.asarray([0.5]), ms.float32)
    output = net(x, y)
    expect_output = Tensor(np.asarray([0.40000004]), ms.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
