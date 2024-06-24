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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import os

import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_adam_bporp_with_cache():
    """
    Feature: cache of bprop expander
    Description: Verify if the loss is converged
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

    class NetAdam(nn.Cell):
        def __init__(self):
            super(NetAdam, self).__init__()
            self.batch_size = 1
            self.reshape = P.Reshape()
            weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
            self.fc1 = nn.Dense(16, 10, weight_init=weight,
                                bias_init="zeros", activation="relu").to_float(ms.float16)
            self.add = P.Add()
            self.cast = P.Cast()

        def construct(self, input_x):
            output = self.reshape(input_x, (self.batch_size, -1))
            output = self.fc1(output)
            output = self.add(output, 0.1)
            output = self.cast(output, ms.float32)
            return output

    def get_loss(net):
        epoch = 3
        optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01)
        criterion = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='mean')
        net_with_criterion = WithLossCell(net, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()

        losses2 = []
        for _ in range(epoch):
            data = Tensor(np.arange(0, 16).reshape(
                (1, 1, 4, 4)).astype(np.float32) * 0.01)
            label = Tensor(np.array([0]).astype(np.int32))
            loss = train_network(data, label)
            losses2.append(loss.asnumpy())
        return losses2

    loss = get_loss(NetAdam())
    os.environ["MS_DEV_DISABLE_BPROP_CACHE"] = "on"
    trace_loss = get_loss(NetAdam())
    os.environ["MS_DEV_DISABLE_BPROP_CACHE"] = "off"
    assert np.allclose(loss, trace_loss, 1e-4, 1e-4)
