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
        return x.H


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_H_complex(mode):
    """
    Feature: tensor.positive
    Description: Verify the result of positive
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-1.5 + 7.8j, 3 + 5.75j]])
    net = Net()
    output = net(x)
    expect_output = [[-1.5000-7.8000j],
                     [3.0000-5.7500j]]
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_positive_01(mode):
    """
    Feature: tensor.positive
    Description: Verify the result of positive
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[12., -51, 4], [6, 167, -68]]), ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[12.0, 6.0],
                     [-51.0, 167.0],
                     [4.0, -68.0]]
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_positive_02(mode):
    """
    Feature: tensor.positive
    Description: Verify the result of positive
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41], [-4, 24, -41]]), ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[12.0, 6., -4.0, -4.0],
                     [-51.0, 167.0, 24.0, 24.0],
                     [4.0, -68.0, -41.0, -41.0]]
    assert np.allclose(output.asnumpy(), expect_output)
