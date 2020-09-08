# Copyright 2020 Huawei Technologies Co., Ltd
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

import sys
import argparse
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.communication.management import init, get_group_size
from mindspore.parallel._ps_context import _is_role_pserver
# from resnet import resnet50

parser = argparse.ArgumentParser(description="test_ps_lenet")
parser.add_argument("--device_target", type=str, default="Ascend")
args, _ = parser.parse_known_args()
device_target = args.device_target
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
context.set_ps_context(enable_ps=True)
if device_target == "GPU":
    init()


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_init=weight,
        has_bias=False,
        pad_mode="valid",
    )


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, channel=3):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    epoch = 5
    np.random.seed(0)
    network = LeNet5(10)
    network.set_param_ps()
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    if device_target == "GPU":
        context.set_auto_parallel_context(parallel_mode="data_parallel", gradients_mean=True,
                                          device_num=get_group_size())
    net_with_criterion = WithLossCell(network, criterion)
    train_network = TrainOneStepCell(net_with_criterion, net_opt)
    train_network.set_train()
    losses = []
    for _ in range(epoch):
        data = Tensor(np.random.rand(32, 3, 32, 32).astype(np.float32))
        label = Tensor(np.random.randint(0, 9, (32)).astype(np.int32))
        if _is_role_pserver():
            train_network(data, label)
            sys.exit()
        else:
            loss = train_network(data, label).asnumpy()
            losses.append(loss)
    print(losses)
