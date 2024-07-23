# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.mint as mint
import mindspore.mint.nn as mnn


class MaxNet(nn.Cell):
    def construct(self, x):
        return mint.max(x, 0, True)


class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.net = mnn.Linear(3, 4)

    def construct(self, x):
        return self.net(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_max_and_linear(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    ms.set_context(mode=mode)
    input0 = Tensor(np.array([[0.0, 0.3, 0.4, 0.5, 0.1], [3.2, 0.4, 0.1, 2.9, 4.0]]), ms.float32)
    input1 = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
    net1 = MaxNet()
    net2 = LinearNet()
    output0, _ = net1(input0)
    output1 = net2(input1).shape

    expect_output0 = np.array([[3.2, 0.4, 0.4, 2.9, 4.]], dtype=np.float32)
    expect_output1 = (2, 4)
    assert np.allclose(output0.asnumpy(), expect_output0)
    assert np.allclose(output1, expect_output1)
