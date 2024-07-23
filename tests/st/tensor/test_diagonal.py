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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def __init__(self, offset=0, axis1=0, axis2=1):
        super(Net, self).__init__()
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def construct(self, x):
        out = x.diagonal(self.offset, self.axis1, self.axis2)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_diagonal(mode):
    """
    Feature: tensor.diagonal
    Description: Verify the result of diagonal
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [0, 3]
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_diagonal_with_zero_shape(mode):
    """
    Feature: tensor.diagonal
    Description: Verify the output of diagonal with zero shape
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[[0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15]],
                   [[16, 17, 18, 19],
                    [20, 21, 22, 23],
                    [24, 25, 26, 27],
                    [28, 29, 30, 31]]], ms.float32)
    net = Net(offset=4, axis1=0, axis2=1)
    output = net(x)
    expect_output = []
    assert output.shape == (4, 0)
    assert np.allclose(output.asnumpy(), expect_output)
