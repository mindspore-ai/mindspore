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
    def construct(self, a, p=None):
        output = ops.cond(a, p)
        return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cond(mode):
    """
    Feature: cond
    Description: Verify the result of cond
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    output = net(x)
    expect_output = np.array(1.4142)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cond_4d(mode):
    """
    Feature: cond
    Description: Verify the result of cond
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor([[[[0.59469778, 0.22015922],
                  [0.69863667, 0.70537429]],
                 [[0.16839681, 0.72470992],
                  [0.29759212, 0.14389902]]],
                [[[0.47768186, 0.07188184],
                  [0.9755139, 0.36467102]],
                 [[0.01699958, 0.30675664],
                  [0.70047389, 0.55853604]]]])
    output = net(x)
    expect_output = np.array([[5.024622, 3.144386], [12.584087, 4.1251683]])
    assert np.allclose(output.asnumpy(), expect_output)
