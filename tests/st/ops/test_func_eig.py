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
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x):
        output = ops.eig(x)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_eig(mode):
    """
    Feature: ops.eig
    Description: Verify the result of ops.eig
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()

    inputs = np.array([[1.0, 0.0], [0.0, 2.0]])

    x = Tensor(inputs, ms.float32)
    value, vector = net(x)
    expect_value = np.array([1.+0.j, 2.+0.j])
    expect_vector = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])
    assert np.allclose(expect_value, value.asnumpy())
    assert np.allclose(expect_vector, vector.asnumpy())
