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
    def construct(self, w, x, y, z):
        output = w.index_put(x, y, z)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_index_put(mode):
    """
    Feature: tensor.index_put
    Description: Verify the result of tensor.index_put
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), ms.int32)
    values = Tensor(np.array([3]), ms.int32)
    indices = [Tensor(np.array([0, 0]).astype(np.int32)), Tensor(np.array([0, 1]).astype(np.int32))]
    accumulate = True
    output = net(x, indices, values, accumulate)
    expect_output = Tensor(np.asarray([[4, 5, 3], [4, 5, 6]]), ms.int32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
