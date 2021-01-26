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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Model, ms_function
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")

input_channel = 2048
output_channel = 512
num_class = 10
batch_size = 32


class MsWrapper(nn.Cell):
    def __init__(self, network):
        super(MsWrapper, self).__init__(auto_prefix=False)
        self._network = network

    @ms_function
    def construct(self, *args):
        return self._network(*args)


def me_train_tensor(net, input_np, label_np, epoch_size=2):
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = nn.Momentum(Tensor(np.array([0.1])), Tensor(np.array([0.9])),
                      filter(lambda x: x.requires_grad, net.get_parameters()))
    context.set_context(mode=context.GRAPH_MODE)
    Model(net, loss, opt)
    _network = nn.WithLossCell(net, loss)
    _train_net = MsWrapper(nn.TrainOneStepCell(_network, opt))
    _train_net.set_train()
    for epoch in range(0, epoch_size):
        print(f"epoch %d" % (epoch))
        output = _train_net(Tensor(input_np), Tensor(label_np))
        print(output.asnumpy())


def test_conv_bn_add_relu_fusion():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(input_channel, output_channel,
                                  kernel_size=1, stride=1, padding=0, has_bias=False, pad_mode="same")
            self.conv1 = nn.Conv2d(input_channel, output_channel,
                                   kernel_size=1, stride=1, padding=0, has_bias=False, pad_mode="same")
            self.bn = nn.BatchNorm2d(output_channel, momentum=0.1, eps=0.0001)
            self.add = P.Add()
            self.relu = P.ReLU()
            self.mean = P.ReduceMean(keep_dims=True)
            self.reshape = P.Reshape()
            self.dense = nn.Dense(output_channel, num_class)

        def construct(self, input_x):
            output = self.conv(input_x)
            output = self.bn(output)
            output = self.add(output, self.conv1(input_x))
            output = self.relu(output)
            output = self.mean(output, (-2, -1))
            output = self.reshape(output, (batch_size, output_channel))
            output = self.dense(output)
            return output

    net = Net()
    input_np = np.ones([batch_size, input_channel, 7, 7]).astype(np.float32) * 0.01
    label_np = np.ones([batch_size]).astype(np.int32)
    me_train_tensor(net, input_np, label_np)


def test_conv_bn_relu_fusion():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(input_channel, output_channel,
                                  kernel_size=1, stride=1, padding=0, has_bias=False, pad_mode="same")
            self.bn = nn.BatchNorm2d(output_channel, momentum=0.1, eps=0.0001)
            self.relu = P.ReLU()
            self.mean = P.ReduceMean(keep_dims=True)
            self.reshape = P.Reshape()
            self.dense = nn.Dense(output_channel, num_class)

        def construct(self, input_x):
            output = self.conv(input_x)
            output = self.bn(output)
            output = self.relu(output)
            output = self.mean(output, (-2, -1))
            output = self.reshape(output, (batch_size, output_channel))
            output = self.dense(output)
            return output

    net = Net()
    input_np = np.ones([batch_size, input_channel, 7, 7]).astype(np.float32) * 0.01
    label_np = np.ones([batch_size]).astype(np.int32)
    me_train_tensor(net, input_np, label_np)


def test_conv_bn_fusion():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(input_channel, output_channel,
                                  kernel_size=1, stride=1, padding=0, has_bias=False, pad_mode="same")
            self.bn = nn.BatchNorm2d(output_channel, momentum=0.1, eps=0.0001)
            self.mean = P.ReduceMean(keep_dims=True)
            self.reshape = P.Reshape()
            self.dense = nn.Dense(output_channel, num_class)

        def construct(self, input_x):
            output = self.conv(input_x)
            output = self.bn(output)
            output = self.mean(output, (-2, -1))
            output = self.reshape(output, (batch_size, output_channel))
            output = self.dense(output)
            return output

    net = Net()
    input_np = np.ones([batch_size, input_channel, 7, 7]).astype(np.float32) * 0.01
    label_np = np.ones([batch_size]).astype(np.int32)
    me_train_tensor(net, input_np, label_np)
