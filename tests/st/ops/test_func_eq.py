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
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x, y):
        output = ops.eq(x, y)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_eq(mode):
    """
    Feature: ops.eq
    Description: Verify the result of ops.eq
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()

    inputs = Tensor(np.array([[1.0, 0.0], [0.0, 2.0]]))
    other = Tensor(np.array([[1.0, 1.0], [1.0, 2.0]]))
    value = net(inputs, other)
    expect_value = np.array([[True, False], [False, True]])
    assert np.allclose(expect_value, value.asnumpy())
