# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
    def construct(self, start=0, end=None, step=1, dtype=None):
        return ops.arange(start, end, step, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_arange_normal(mode):
    """
    Feature: arange
    Description: Verify the result of arange
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    output1 = net(1, 6)
    output2 = net(10.0, dtype=ms.int32)
    output3 = net(ms.Tensor(12.0, dtype=ms.float64), 2, ms.Tensor(-1.0, dtype=ms.float32))
    assert np.allclose(output1.asnumpy(), np.array([1., 2., 3., 4., 5.]))
    assert output1.dtype == ms.int64
    assert np.allclose(output2.asnumpy(), np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))
    assert output2.dtype == ms.int32
    assert np.allclose(output3.asnumpy(), np.array([12., 11., 10., 9., 8., 7., 6., 5., 4., 3.]))
    assert output3.dtype == ms.float32
