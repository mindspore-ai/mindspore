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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, dtype):
        out = ops.zeros_like(x, dtype=dtype)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, ms.int32])
def test_zeros_like(mode, dtype):
    """
    Feature: ops.zeros_like
    Description: Verify the result of zeros_like
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.arange(9).reshape((3, 3)))
    net = Net()
    output = net(x, dtype)
    if dtype is None:
        assert output.dtype == x.dtype
    else:
        assert output.dtype == dtype
    expect_out = np.zeros((3, 3))
    assert np.array_equal(output.asnumpy(), expect_out)
