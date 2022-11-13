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
from mindspore import Tensor, ops


class NetWorkClipByValue(nn.Cell):
    def construct(self, x, min_value, max_value):
        return ops.clip_by_value(x, min_value, max_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clip_by_value_tensor(mode):
    """
    Feature: ops.clip_by_value
    Description: Verify the result of clip_by_value
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    net = NetWorkClipByValue()
    output = net(x, -0.3, 0.4)
    expect_output = [-0.3, 0.4, 0.2349, -0.3, 0.4]
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clip_by_value_list_tensor(mode):
    """
    Feature: ops.clip_by_value
    Description: Verify the result of clip_by_value
    Expectation: success
    """
    ms.set_context(mode=mode)
    x1 = Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    x2 = Tensor(np.array([0.6035, 0.6959, 0.0150, -0.5766, 0.5432]), ms.float32)
    x3 = Tensor(np.array([0.7549, 0.1056, 0.3312, -0.4060, 0.9821]), ms.float32)
    net = NetWorkClipByValue()
    output = net([x1, x2, x3], -0.3, 0.4)
    expect_output = [[-0.3, 0.4, 0.2349, -0.3, 0.4],
                     [0.4, 0.4, 0.0150, -0.3, 0.4],
                     [0.4, 0.1056, 0.3312, -0.3, 0.4]
                     ]
    assert np.allclose(output[0].asnumpy(), expect_output[0])
    assert np.allclose(output[1].asnumpy(), expect_output[1])
    assert np.allclose(output[2].asnumpy(), expect_output[2])
