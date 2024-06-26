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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn


class Argsort(nn.Cell):
    def construct(self, input_x, axis=-1, descending=False):
        return input_x.argsort(axis, descending)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_argsort(mode):
    """
    Feature: tensor.argsort
    Description: Verify the result of argsort
    Expectation: success
    """
    ms.set_context(mode=mode)
    a = [[0.0785, 1.5267, -0.8521, 0.4065],
         [0.1598, 0.0788, -0.0745, -1.2700],
         [1.2208, 1.0722, -0.7064, 1.2564],
         [0.0669, -0.2318, -0.8229, -0.9280]]
    x = ms.Tensor(a)
    net = Argsort()
    out = net(x)
    expect = [[2, 0, 3, 1],
              [3, 2, 1, 0],
              [2, 1, 0, 3],
              [3, 2, 1, 0]]
    assert np.allclose(out.asnumpy(), np.array(expect))
