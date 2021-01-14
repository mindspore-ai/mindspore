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

import numpy as np

import mindspore as ms
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import ReLU
from mindspore.nn.layer.conv import Conv2d
from mindspore.nn.layer.normalization import BatchNorm2d
from mindspore.nn.layer.pooling import MaxPool2d
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData

dev_num = 8
strategy_weight = ((dev_num, 1, 1, 1), (1, 1, 1, 1))
strategy_bn = ((dev_num, 1, 1, 1), (1,), (1,), (1,), (1,))
strategy_fc_weight_bias = ((dev_num, 1), (1, 1), (1,))


class DatasetLenet(MindData):
    def __init__(self, predict, label, length=3):
        super(DatasetLenet, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


def conv7x7(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    weight_shape = (out_channels, in_channels, 7, 7)
    weight = Tensor(np.ones(weight_shape).astype(np.float32))
    conv = Conv2d(in_channels, out_channels,
                  kernel_size=7, stride=stride, padding=padding, weight_init=weight, has_bias=False,
                  pad_mode="same")
    conv.conv2d.shard(strategy_weight)
    return conv


def weight_variable_0(shape):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def weight_variable_1(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def bn_with_initialize(out_channels):
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    gamma = weight_variable_1(shape)
    bn = BatchNorm2d(out_channels, momentum=0.1, eps=0.0001, gamma_init=gamma,
                     beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    bn.bn_train.shard(strategy_bn)
    return bn


class ResNet(Cell):

    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        strategy_no_weight = ((dev_num, 1, 1, 1),)
        self.conv1 = conv7x7(3, 64, stride=2, padding=0)
        self.bn1 = bn_with_initialize(64)
        self.relu = ReLU()
        self.relu.relu.shard(strategy_no_weight)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(((8, 1), (1, 1)))
        self.matmul_weight = Parameter(Tensor(np.ones([200704, num_classes]), dtype=ms.float32), name="weight")

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.reshape(x, (32 * dev_num, 200704))
        x = self.matmul(x, self.matmul_weight)
        return x


def batchnorm_net(num_classes):
    return ResNet(num_classes)


def test_batchnorm_batch_parallel():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=dev_num)
    context.set_context(mode=context.GRAPH_MODE)
    num_classes = 1001
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    predict = Tensor(np.ones([batch_size, 3, 224, 224]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)

    dataset = DatasetLenet(predict, label, 2)
    net = batchnorm_net(num_classes)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss.softmax_cross_entropy.shard(((dev_num, 1), (dev_num, 1)))
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)

    model = Model(net, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


if __name__ == '__main__':
    test_batchnorm_batch_parallel()
