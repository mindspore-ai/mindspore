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
import datetime
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_group_size
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
init()

epoch = 5
total = 5000
batch_size = 32
mini_batch = total // batch_size


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()

        self.relu = P.ReLU()
        self.batch_size = 32
        weight1 = Tensor(np.ones([6, 3, 5, 5]).astype(np.float32) * 0.01)
        weight2 = Tensor(np.ones([16, 6, 5, 5]).astype(np.float32) * 0.01)
        self.conv1 = nn.Conv2d(3, 6, (5, 5), weight_init=weight1, stride=1, padding=0, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, (5, 5), weight_init=weight2, pad_mode='valid', stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")
        self.reshape = P.Reshape()

        weight1 = Tensor(np.ones([120, 400]).astype(np.float32) * 0.01)
        self.fc1 = nn.Dense(400, 120, weight_init=weight1)

        weight2 = Tensor(np.ones([84, 120]).astype(np.float32) * 0.01)
        self.fc2 = nn.Dense(120, 84, weight_init=weight2)

        weight3 = Tensor(np.ones([10, 84]).astype(np.float32) * 0.01)
        self.fc3 = nn.Dense(84, 10, weight_init=weight3)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def test_lenet_nccl():
    context.set_auto_parallel_context(parallel_mode="data_parallel", gradients_mean=True, device_num=get_group_size())
    net = LeNet()
    net.set_train()

    learning_rate = 0.01
    momentum = 0.9
    mom_optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, mom_optimizer)
    train_network.set_train()
    losses = []

    data = Tensor(np.ones([net.batch_size, 3, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([net.batch_size]).astype(np.int32))
    start = datetime.datetime.now()
    for _ in range(epoch):
        for _ in range(mini_batch):
            loss = train_network(data, label)
            losses.append(loss.asnumpy())
    end = datetime.datetime.now()
    with open("ms_time.txt", "w") as fo1:
        fo1.write("time:")
        fo1.write(str(end - start))
    with open("ms_loss.txt", "w") as fo2:
        fo2.write("loss:")
        fo2.write(str(losses[-5:]))
    assert losses[-1] < 0.01
