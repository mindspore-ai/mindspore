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
"""
Function:
    test network
Usage:
    python test_network_main.py --net lenet --target Ascend
"""
import argparse

import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum

from .models.alexnet import AlexNet
from .models.lenet import LeNet
from .models.resnetv1_5 import resnet50


def train(net, data, label):
    learning_rate = 0.01
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    res = train_network(data, label)
    print(res)
    assert res


def test_resnet50():
    data = Tensor(np.ones([32, 3, 224, 224]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = resnet50(32, 10)
    train(net, data, label)


def test_lenet():
    net = LeNet()
    data = Tensor(np.ones([net.batch_size, 3, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([net.batch_size]).astype(np.int32))
    train(net, data, label)


def test_alexnet():
    data = Tensor(np.ones([32, 3, 227, 227]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = AlexNet()
    train(net, data, label)


parser = argparse.ArgumentParser(description='MindSpore Testing Network')
parser.add_argument('--net', default='resnet50', type=str, help='net name')
parser.add_argument('--device', default='Ascend', type=str, help='device target')
if __name__ == "__main__":
    args = parser.parse_args()
    context.set_context(device_target=args.device)
    if args.net == 'resnet50':
        test_resnet50()
    elif args.net == 'lenet':
        test_lenet()
    elif args.net == 'alexnet':
        test_alexnet()
    else:
        print("Please add net name like --net lenet")
