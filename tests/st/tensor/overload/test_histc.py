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

import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class HistcNet(nn.Cell):
    def construct(self, x, bins=100, min_val=0, max_val=0):
        return x.histc(bins, min_val, max_val)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_histc(mode):
    """
    Feature: Tensor.histc
    Description: Verify the result of Tensor.histc
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = Tensor([2, 4, 1, 0, 0], ms.int32)
    bins = 5
    min_val = 0
    max_val = 5
    net = HistcNet()
    output = net(x, bins, min_val, max_val)
    expect_output = Tensor([2, 1, 1, 0, 1], ms.int32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
