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
    # pylint: disable=redefined-builtin
    def construct(self, input_x, k, dim=None, sorted=True):
        output = ops.top_k(input_x, k, dim=dim, sorted=sorted)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_top_k_normal(mode):
    """
    Feature: top_k
    Description: Verify the result of top_k
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[0.5368, 0.2447, 0.4302, 0.9673],
                   [0.4388, 0.6525, 0.4685, 0.1868],
                   [0.3563, 0.5152, 0.9675, 0.8230]], dtype=ms.float32)
    output = net(x, 2, dim=1)
    output0 = output[0]
    output1 = output[1]
    expect_output0 = np.array([[0.9673, 0.5368],
                               [0.6525, 0.4685],
                               [0.9675, 0.823]])
    expect_output1 = np.array([[3, 0],
                               [1, 2],
                               [2, 3]])
    assert np.allclose(output0.asnumpy(), expect_output0)
    assert np.allclose(output1.asnumpy(), expect_output1)
