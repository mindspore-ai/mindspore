# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.nn import TrainOneStepCell, WithLossCell
from tests.st.networks.models.lenet import LeNet

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lenet():
    '''
    Feature: AdaFactor
    Description: Test AdaFactor
    Expectation: Run lenet success
    '''
    data = Tensor(np.ones([32, 3, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = LeNet()
    net.batch_size = 32
    learning_rate = 0.01
    optimizer = nn.AdaFactor(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate,
                             scale_parameter=False, relative_step=False, beta1=0)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    loss = []
    for _ in range(10):
        res = train_network(data, label)
        loss.append(res.asnumpy())
    assert np.all(loss[-1] < 0.1)
