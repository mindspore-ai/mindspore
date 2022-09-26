# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class MomentumNet(nn.Cell):
    def __init__(self):
        super(MomentumNet, self).__init__()
        self.batch_size = 1

        self.reshape = P.Reshape()
        weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        self.fc1 = Dense(16, 10, weight_init=weight)

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_momentum():
    epoch = 13
    net = MomentumNet()
    learning_rate = 0.1
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    losses = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape(1, 1, 4, 4).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss)

    print("================================")
    print(losses)
#   expect output:
#   [[0.04132498 0.00874167 0.00874167 0.00874167 0.00874167
#     0.00874167 0.00874167 0.00874167 0.00874167 0.00874167]]

    return losses


class DynamicShapeNet(nn.Cell):
    def __init__(self, variable, accumulation):
        super(DynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.variable = Parameter(variable, name="variable")
        self.accumulation = Parameter(accumulation, name="accumulation")
        self.apply_momentum = P.ApplyMomentum()

    def construct(self, learning_rate, gradient, momentum, indices):
        unique_indices, _ = self.unique(indices)
        gradient = self.gather(gradient, unique_indices, 0)
        return self.apply_momentum(self.variable, self.accumulation, learning_rate, gradient, momentum)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_momentum_dynamic_shape():
    """
    Feature: test ApplyMomentum dynamic_shape feature.
    Description: test ApplyMomentum dynamic_shape feature.
    Expectation: success.
    """
    # dynamic inputs
    indices_np = np.random.randint(0, 3, size=6)
    indices_ms = Tensor(indices_np)

    # data preparation
    unique_indices, _ = P.Unique()(indices_ms)
    variable = Tensor(np.arange(20).reshape(4, 5).astype(np.float32) / 10)
    variable = P.Gather()(variable, unique_indices, 0)
    accumulation = Tensor(np.arange(20).reshape(4, 5).astype(np.float32) / 10)
    accumulation = P.Gather()(accumulation, unique_indices, 0)
    learning_rate = Tensor(np.array([0.0001]).astype(np.float32))
    gradient = Tensor(np.arange(24, 44).reshape(4, 5).astype(np.float32))
    momentum = Tensor(np.array([0.0001]).astype(np.float32))

    # dynamic shape
    gradient_dyn = Tensor(shape=[None for _ in gradient.shape], dtype=gradient.dtype)
    dynamic_shape_net = DynamicShapeNet(variable, accumulation)
    dynamic_shape_net.set_inputs(learning_rate, gradient_dyn, momentum, indices_ms)

    # run in graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    outputs = dynamic_shape_net(learning_rate, gradient, momentum, indices_ms)
    expect_shape = variable.asnumpy().shape
    assert outputs.asnumpy().shape == expect_shape
