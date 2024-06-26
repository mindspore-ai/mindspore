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
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, condition, x, y):
        return x.where(condition, y)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_where(mode):
    """
    Feature: tensor.where
    Description: Verify the result of where
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.arange(4).reshape((2, 2)), mstype.float32)
    y = Tensor(np.ones((2, 2)), mstype.float32)
    condition = x < 3
    net = Net()
    output = net(condition, x, y)
    expected = np.array([[0, 1], [2, 1]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
