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
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x, rep, axis):
        output = ops.repeat_elements(x, rep, axis)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_repeat_elements(mode):
    """
    Feature: real
    Description: Verify the result of repeat_elements when axis is less than 0.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.array([[0, 1, 2], [3, 4, 5]]), ms.int32)
    out = net(x, 2, -1)
    expect_out = np.array([[0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5]])
    assert np.allclose(out.asnumpy(), expect_out)
    out = net(x, 2, -2)
    expect_out = np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]])
    assert np.allclose(out.asnumpy(), expect_out)
    x = ms.Tensor(np.random.randn(1, 2, 3, 4, 5, 6, 7, 8), ms.float32)
    with pytest.raises(ValueError,
                       match="For repeat_elements, the rank of x should be less than or equal to 7, but got 8."):
        _ = net(x, 1, 0)
        _pynative_executor.sync()
