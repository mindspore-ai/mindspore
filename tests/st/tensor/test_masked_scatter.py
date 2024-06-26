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
    def construct(self, x, y, z):
        output = x.masked_scatter(y, z)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_masked_scatter(mode):
    """
    Feature: tensor.masked_scatter
    Description: Verify the result of tensor.masked_scatter
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([1., 2., 3., 4.]), ms.float32)
    mask = Tensor(np.array([True, True, False, True]), ms.bool_)
    tensor = Tensor(np.array([5., 6., 7.]), ms.float32)
    output = net(x, mask, tensor)
    expect_output = Tensor(np.asarray([5., 6., 3., 7.]), ms.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
