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
<<<<<<< HEAD:tests/st/networks/test_network_main.py
"""
Function: 
    test network
Usage: 
    python test_network_main.py --net lenet --target Ascend
"""
import os
import time
import numpy as np
import argparse
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
import mindspore.context as context
from mindspore.nn.optim import Momentum
from models.lenet import LeNet
from models.resnetv1_5 import resnet50
from models.alexnet import AlexNet
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
=======
import pytest
from mindspore.nn import TrainOneStepCell, WithLossCell
import mindspore.context as context
from mindspore.nn.optim import Momentum
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor

class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 32

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
>>>>>>> add cpu st lenet:tests/st/networks/test_cpu_lenet.py

def train(net, data, label):
    learning_rate = 0.01
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    res = train_network(data, label)
    print("+++++++++Loss+++++++++++++")
    print(res)
    print("+++++++++++++++++++++++++++")
    assert res

<<<<<<< HEAD:tests/st/networks/test_network_main.py
def test_resnet50():
    data = Tensor(np.ones([32, 3 ,224, 224]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = resnet50(32, 10)
    train(net, data, label)

=======
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
>>>>>>> add cpu st lenet:tests/st/networks/test_cpu_lenet.py
def test_lenet():
    data = Tensor(np.ones([32, 1 ,32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = LeNet()
    train(net, data, label)
<<<<<<< HEAD:tests/st/networks/test_network_main.py

def test_alexnet():
    data = Tensor(np.ones([32, 3 ,227, 227]).astype(np.float32) * 0.01)
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
=======
>>>>>>> add cpu st lenet:tests/st/networks/test_cpu_lenet.py
