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
    def construct(self, dtype):
        out = ops.zeros((3, 4, 5), dtype=dtype)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, ms.int32])
def test_zeros(mode, dtype):
    """
    Feature: ops.zeros
    Description: Verify the result of ops.zeros
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    output = net(dtype)
    if dtype is None:
        assert output.dtype == ms.float32
    else:
        assert output.dtype == dtype
    expect_out = np.zeros((3, 4, 5))
    assert np.array_equal(output.asnumpy(), expect_out)
