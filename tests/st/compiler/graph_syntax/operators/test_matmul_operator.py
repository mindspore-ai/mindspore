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
""" test matmul operator '@' """
import numpy as np
import pytest
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_matmul_operator():
    """
    Feature: operator '@' for tensor
    Description: test '@' operator between Tensor and Tensor
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x1, y1):
            x2 = Tensor(np.array([1, 2]), mindspore.float32)
            y2 = Tensor(np.array([1, 2]).T, mindspore.float32)
            return x1 @ y1, x2 @ y2

    x_np = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype('float32')
    y_np = np.arange(4 * 5).reshape(4, 5).astype('float32')
    x = Tensor(x_np)
    y = Tensor(y_np)
    net = Net()
    result1, result2 = net(x, y)
    assert np.allclose(result1.asnumpy(), x_np @ y_np)
    assert result2 == 5
