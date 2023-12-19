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
from mindspore import ops


class NetAdam(nn.Cell):
    def __init__(self):
        super(NetAdam, self).__init__()
        self.batch_size = 1
        self.reshape = ops.Reshape()
        weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        self.fc1 = nn.Dense(16, 10, weight_init=weight, bias_init='zeros')

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_trainonestepcellwithgrad(mode):
    """
    Feature: trainonestepcellwithgrad
    Description: Verify the trainonestepcell can return dict type grad
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NetAdam()
    optimizer = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = nn.WithLossCell(net, criterion)
    train_network = nn.TrainOneStepCell(net_with_criterion, optimizer, return_grad=True)
    train_network.set_train()

    data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
    label = Tensor(np.array([0]).astype(np.int32))
    _, grads = train_network(data, label)
    assert isinstance(grads, dict)
