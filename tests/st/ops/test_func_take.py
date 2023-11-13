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
    def construct(self, x, indices, axis=None, mode='clip'):
        return ops.take(x, indices, axis, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_take_none_axis(mode):
    """
    Feature: take
    Description: Verify the result of take when axis is None
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.array([4, 3, 5, 7, 6, 8]))
    indices = ms.Tensor(np.array([0, 1, 4]))
    output = net(x, indices)
    expect_output = np.array([4, 3, 6])
    assert np.allclose(output.asnumpy(), expect_output)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_take_with_axis(mode):
    """
    Feature: take
    Description: Verify the result of take when axis is not None
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.array([[4, 3, 5], [7, 6, 8]]))
    indices = ms.Tensor(np.array([0, 2]))
    output = net(x, indices, 1)
    expect_output = np.array([[4, 5], [7, 8]])
    assert np.allclose(output.asnumpy(), expect_output)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_take_with_raise_mode(mode):
    """
    Feature: take
    Description: Verify the illegal input for take when mode set to `raise`
    Expectation: excption
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.array([[4, 3, 5], [7, 6, 8]]))
    indices = ms.Tensor(np.array([0, 3]))
    with pytest.raises(ValueError) as err:
        _ = net(x, indices, 1, mode='raise')
    assert "indice out of range" in str(err.value)
