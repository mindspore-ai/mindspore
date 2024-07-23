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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from tests.mark_utils import arg_mark


class MarginRankingLoss(nn.Cell):
    def __init__(self, reduction="none"):
        super(MarginRankingLoss, self).__init__()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=0.0, reduction=reduction)

    def construct(self, x, y, label):
        return self.margin_ranking_loss(x, y, label)


input1 = Tensor(np.array([0.3864, -2.4093, -1.4076]), ms.float32)
input2 = Tensor(np.array([-0.6012, -1.6681, 1.2928]), ms.float32)
target = Tensor(np.array([-1, -1, 1]), ms.float32)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ["none", "mean", "sum"])
def test_margin_ranking_loss(mode, reduction):
    """
    Feature: test MarginRankingLoss op with reduction none.
    Description: Verify the result of MarginRankingLoss.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    loss = MarginRankingLoss(reduction=reduction)
    output = loss(input1, input2, target)
    if reduction == 'none':
        expect_output = np.array([0.98759997, 0., 2.7003999])
    elif reduction == 'sum':
        expect_output = np.array(3.6879997)
    else:
        expect_output = np.array(1.2293333)

    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_tensor_dim(mode):
    """
    Feature: test tensor dim
    Description: Verify the result of dim.
    Expectation: expect correct forward result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        def construct(self, x):
            return x.dim(), self.tensor.dim()

    net = Net()
    input11 = Tensor([[1, 2, 3], [4, 5, 6]])
    input22 = Tensor([[[1, 2, 3], [4, 5, 6]]])
    net(input11)
    net(input22)
