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
import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, y):
        output = x.outer(y)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_outer(mode):
    """
    Feature: tensor.outer
    Description: Verify the result of tensor.outer
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x1 = Tensor(np.array([7, 8, 9]), ms.int32)
    x2 = Tensor(np.array([7, 10, 11]), ms.int32)
    output = net(x1, x2)
    expect_output = Tensor(np.asarray([[49, 70, 77], [56, 80, 88], [63, 90, 99]]), ms.int32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
