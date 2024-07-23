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

import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def __init__(self, index=None):
        super(Net, self).__init__()
        self.index = index

    def construct(self, x):
        if self.index is None:
            return x.item()
        return x.item(self.index)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_item(mode):
    """
    Feature: tensor.item
    Description: Verify the result of item
    Expectation: success
    """
    ms.set_context(mode=mode)
    eps = 1e-6
    x = ms.Tensor(1.2, ms.float32)
    net = Net()
    output = net(x)
    expect_output = 1.2
    assert abs(output - expect_output) <= eps

    x = ms.Tensor([2.98], ms.float32)
    net = Net()
    output = net(x)
    expect_output = 2.98
    assert abs(output - expect_output) <= eps

    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
    net = Net((0, 1))
    output = net(x)
    expect_output = 2.0
    assert abs(output - expect_output) <= eps
