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

from ssl_args import ClusterArgs
import mindspore.context as context
import mindspore.nn as nn
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_group_size
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P

cluster_args = ClusterArgs(description="config SSL for CPU data parallel")
enable_ssl = cluster_args.enable_ssl
config_file_path = cluster_args.config_file_path
client_password = cluster_args.client_password
server_password = cluster_args.server_password
security_ctx = {
    "enable_ssl": enable_ssl,
    "config_file_path": config_file_path,
    "client_password": client_password,
    "server_password": server_password
}

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
context.set_ps_context(**security_ctx)
init()
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                  gradients_mean=True,
                                  device_num=get_group_size())


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
        self.flatten = nn.Flatten()

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
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def multi_step_lr(total_steps, gap, base_lr=0.9, gamma=0.1, dtype=mstype.float32):
    lr = []
    for step in range(total_steps):
        lr_ = base_lr * gamma ** (step // gap)
        lr.append(lr_)
    return Tensor(np.array(lr), dtype)


def train_lenet_by_cpu_data_parallel():
    """Train lenet by cpu data parallel"""
    epoch = 10
    total = 5000
    batch_size = 32
    step = total // batch_size

    net = LeNet()
    learning_rate = multi_step_lr(epoch, 2)
    momentum = 0.9
    mom_optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, mom_optimizer)
    train_network.set_train()
    assert train_network.parallel_mode == context.ParallelMode.DATA_PARALLEL
    assert isinstance(train_network.grad_reducer, nn.DistributedGradReducer)
    assert hasattr(train_network.grad_reducer, "allreduce")

    data = Tensor(np.ones([net.batch_size, 3, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([net.batch_size]).astype(np.int32))
    loss = 1.0
    for _ in range(epoch):
        for _ in range(step):
            loss = train_network(data, label)
            if loss < 0.01:
                return
    assert loss < 0.01


train_lenet_by_cpu_data_parallel()
