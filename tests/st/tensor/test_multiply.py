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
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, y):
        return x.multiply(y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_multiply(mode):
    """
    Feature: test Tensor.log10.
    Description: Verify the result of Tensor.log10.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor([1, 2, 3], dtype=ms.float32)
    y = Tensor([1, 2, 3], dtype=ms.float32)
    multiply = Net()
    output = multiply(x, y)
    expect_output = np.array([1, 4, 9], dtype=np.float32)

    assert np.allclose(output.asnumpy(), expect_output)
