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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    # pylint: disable=redefined-builtin
    def construct(self, input_x, k, dim=None, largest=True, sorted=True):
        output = input_x.topk(k, dim=dim, largest=largest, sorted=sorted)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_topk_normal(mode):
    """
    Feature: top_k
    Description: Verify the result of topk
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
    output2 = net(x, 2, dim=1, largest=False)
    output2_0 = output2[0]
    output2_1 = output2[1]
    expect_output2_0 = np.array([[2.44700000e-01, 4.30200011e-01],
                                 [1.86800003e-01, 4.38800007e-01],
                                 [3.56299996e-01, 5.15200019e-01]])
    expect_output2_1 = np.array([[1, 2],
                                 [3, 0],
                                 [0, 1]])
    assert np.allclose(output0.asnumpy(), expect_output0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output1.asnumpy(), expect_output1, rtol=1e-3, atol=1e-5)
    assert np.allclose(output2_0.asnumpy(), expect_output2_0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output2_1.asnumpy(), expect_output2_1, rtol=1e-3, atol=1e-5)
