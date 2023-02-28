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

'''test overflow'''
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, Parameter, nn, ops, boost
from mindspore import dtype as mstype


class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.full([in_features, out_features], 2, np.float16)),
                                name='weight')
        self.matmul = ops.MatMul()

    def construct(self, x):
        output = self.matmul(x, self.weight)
        return output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_train_one_step_with_loss_scale_cell_overflow(mode):
    """
    Feature: mindspore.TrainOneStepWithLossScaleCell.overflow
    Description: test TrainOneStepWithLossScaleCell overflow
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    size, in_features, out_features = 1, 2, 2
    net = Net(in_features, out_features)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(),
                            learning_rate=0.1, momentum=0.9)
    net_with_loss = nn.WithLossCell(net, loss)
    shape = (size, in_features)
    inputs = [
        Tensor(np.full(shape, -np.inf, np.float16)),
        Tensor(np.full(shape, 0, np.float16)),
        Tensor(np.full(shape, 40000, np.float16)),
        Tensor(np.full(shape, 10, np.float16)),
        Tensor(np.full(shape, np.inf, np.float16)),
    ]
    label = Tensor(np.full([out_features,], 0, np.float16))
    datasets = list(zip(inputs, [label for _ in range(len(inputs))]))
    scaling_sens = Tensor([8], dtype=mstype.float16)
    train_network = nn.TrainOneStepWithLossScaleCell(
        net_with_loss, optimizer, scale_sense=scaling_sens)
    expect_results = [True, False, True, False, True]
    outputs = []
    for x, label in datasets:
        _, overflow, _ = train_network(x, label)
        outputs.append(overflow.asnumpy().tolist())
    assert outputs == expect_results


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_boost_train_one_step_with_loss_scale_cell_overflow(mode):
    """
    Feature: mindspore.BoostTrainOneStepWithLossScaleCell.overflow
    Description: test BoostTrainOneStepWithLossScaleCell overflow
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    size, in_features, out_features = 1, 2, 2
    net = Net(in_features, out_features)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(),
                            learning_rate=0.1, momentum=0.9)
    net_with_loss = nn.WithLossCell(net, loss)
    shape = (size, in_features)
    inputs = [
        Tensor(np.full(shape, -np.inf, np.float16)),
        Tensor(np.full(shape, 0, np.float16)),
        Tensor(np.full(shape, 40000, np.float16)),
        Tensor(np.full(shape, 10, np.float16)),
        Tensor(np.full(shape, np.inf, np.float16)),
    ]
    label = Tensor(np.full([out_features,], 0, np.float16))
    datasets = list(zip(inputs, [label for _ in range(len(inputs))]))
    scaling_sens = Tensor([8], dtype=mstype.float16)
    train_network = boost.BoostTrainOneStepWithLossScaleCell(
        net_with_loss, optimizer, scale_sense=scaling_sens)
    expect_results = [True, False, True, False, True]
    outputs = []
    for x, label in datasets:
        _, overflow, _ = train_network(x, label)
        outputs.append(overflow)
    assert outputs == expect_results
