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


class NetIndexBool(nn.Cell):
    def construct(self, x):
        x[True] *= 3
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_index_bool(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with indexes are bool
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NetIndexBool()
    x = Tensor(np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32))
    x = net(Tensor(x))
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    x_np[True] *= 3
    assert np.allclose(x_np, x.asnumpy())


class NetIndexNone(nn.Cell):
    def construct(self, x):
        x[None] *= 3
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_index_none(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with indexes are None
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NetIndexBool()
    x = Tensor(np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32))
    x = net(Tensor(x))
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    x_np[None] *= 3
    assert np.allclose(x_np, x.asnumpy())
