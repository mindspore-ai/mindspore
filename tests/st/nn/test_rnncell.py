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
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.rnncell = nn.RNNCell(10, 16, dtype=ms.float16)

    def construct(self, x, y):
        out = self.rnncell(x, y)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rnncell_para_customed_dtype(mode):
    """
    Feature: RNNCell
    Description: Verify the result of RNNCell specifying customed para dtype.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.ones([5, 3, 10]).astype(np.float16))
    hx = ms.Tensor(np.ones([3, 16]).astype(np.float16))
    output = []
    for i in range(5):
        hx = net(x[i], hx)
        output.append(hx)
    expect_output_shape = (3, 16)
    assert np.allclose(expect_output_shape, output[0].shape)
