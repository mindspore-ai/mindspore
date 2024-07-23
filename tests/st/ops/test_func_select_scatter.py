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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops


class Net(nn.Cell):
    def __init__(self, axis, index):
        super(Net, self).__init__()
        self.axis = axis
        self.index = index

    def construct(self, input_x, src):
        return ops.select_scatter(input_x, src, self.axis, self.index)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_select_scatter(mode):
    """
    Feature: ops.select_scatter
    Description: Verify the result of select_scatter
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]],
                [[9, 10, 11],
                 [12, 13, 14],
                 [15, 16, 17]]], ms.float32)
    y = Tensor([[18, 19, 20],
                [21, 22, 23]], ms.float32)
    net = Net(1, 0)
    output = net(x, y)
    expect_output = [[[18., 19., 20.],
                      [3., 4., 5.],
                      [6., 7., 8.]],
                     [[21., 22., 23.],
                      [12., 13., 14.],
                      [15., 16., 17.]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(1, -1)
    output = net(x, y)
    expect_output1 = [[[0., 1., 2.],
                       [3., 4., 5.],
                       [18., 19., 20.]],
                      [[9., 10., 11.],
                       [12., 13., 14.],
                       [21., 22., 23.]]]
    assert np.allclose(output.asnumpy(), expect_output1)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_select_scatter_error(mode):
    """
    Feature: ops.select_scatter error
    Description: Verify error of select_scatter
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]],
                [[9, 10, 11],
                 [12, 13, 14],
                 [15, 16, 17]]], ms.float32)
    y = Tensor([[18, 19, 20]], ms.float32)
    net = Net(2, 0)
    with pytest.raises(ValueError):
        net(x, y)

    with pytest.raises(TypeError):
        net(1., 2)
