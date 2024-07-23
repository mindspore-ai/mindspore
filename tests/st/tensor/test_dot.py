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
        output = x.dot(y)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_qr(mode):
    """
    Feature: tensor.dot
    Description: Verify the result of tensor.dot
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    input_x = Tensor(np.ones(shape=[2, 3]), ms.float32)
    other = Tensor(np.ones(shape=[1, 3, 2]), ms.float32)
    output = net(input_x, other)
    output_shape = Tensor(output.shape)
    expect_output = Tensor(np.asarray([[3., 3.], [3., 3.]]), ms.float32)
    expect_output_shape = Tensor(np.asarray([2, 1, 2]), ms.int32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
    assert np.allclose(output_shape.asnumpy(), expect_output_shape.asnumpy())
