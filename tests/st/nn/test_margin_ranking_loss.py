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


class MarginRankingLoss(nn.Cell):
    def __init__(self, reduction="none"):
        super(MarginRankingLoss, self).__init__()
        self.margin_ranking_loss = nn.MarginRankingLoss(reduction=reduction)

    def construct(self, x, y, label):
        return self.margin_ranking_loss(x, y, label)


input1 = Tensor(np.array([0.3864, -2.4093, -1.4076]), ms.float32)
input2 = Tensor(np.array([-0.6012, -1.6681, 1.2928]), ms.float32)
target = Tensor(np.array([-1, -1, 1]), ms.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_margin_ranking_loss_none(mode):
    """
    Feature: test MarginRankingLoss op with reduction none.
    Description: Verify the result of MarginRankingLoss.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    loss = MarginRankingLoss('none')
    output = loss(input1, input2, target)
    expect_output = np.array([0.98759997, 0., 2.7003999])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_margin_ranking_loss_sum(mode):
    """
    Feature: test MarginRankingLoss op with reduction sum.
    Description: Verify the result of MarginRankingLoss.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    loss = MarginRankingLoss('sum')
    output = loss(input1, input2, target)
    expect_output = np.array(3.6879997)
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_margin_ranking_loss_mean(mode):
    """
    Feature: test MarginRankingLoss op with reduction mean.
    Description: Verify the result of MarginRankingLoss.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    loss = MarginRankingLoss('mean')
    output = loss(input1, input2, target)
    expect_output = np.array(1.2293333)
    assert np.allclose(output.asnumpy(), expect_output)



@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
