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
import mindspore.ops.operations.math_ops as M
from mindspore import context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.nn.layer.basic import Flatten
from mindspore.nn.layer.conv import Conv2d
from mindspore.nn.layer.normalization import BatchNorm2d
from mindspore.nn.layer.pooling import MaxPool2d
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.ops.operations import Add
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData

dev_num = 8
strategy_no_weight = ((dev_num, 1, 1, 1),)
strategy_weight = ((dev_num, 1, 1, 1), (1, 1, 1, 1))
strategy_add = ((dev_num, 1, 1, 1), (dev_num, 1, 1, 1))
strategy_bn = ((dev_num, 1, 1, 1), (1,), (1,), (1,), (1,))

strategy_fc_weight_nobias = ((1, dev_num), (1, dev_num))
strategy_tensor_add = ((1, dev_num), (dev_num,))


class DenseWrap(Cell):
    def __init__(self,
                 input_channels,
                 output_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 matmul_strategy=None,
                 shard=None):

        super(DenseWrap, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_bias = has_bias

        self.weight = Parameter(initializer(
            weight_init, [output_channels, input_channels]), name="weight")

        if self.has_bias:
            self.bias = Parameter(initializer(
                bias_init, [output_channels]), name="bias")
        else:
            self.bias = Parameter(initializer(
                'zeros', [output_channels]), name="bias")

        self.matmul = P.MatMul(transpose_b=True).shard(matmul_strategy)
        self.bias_add = P.Add().shard(shard)

    def construct(self, x):
        if self.has_bias:
            output = self.bias_add(self.matmul(x, self.weight), self.bias)
        else:
            output = self.matmul(x, self.weight)
        return output


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


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution """
    weight_shape = (out_channels, in_channels, 3, 3)
    weight = Tensor(np.ones(weight_shape).astype(np.float32))
    conv = Conv2d(in_channels, out_channels,
                  kernel_size=3, stride=stride, padding=0, weight_init=weight, has_bias=False,
                  pad_mode="same")
    conv.conv2d.shard(strategy_weight)
    return conv


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    weight_shape = (out_channels, in_channels, 1, 1)
    weight = Tensor(np.ones(weight_shape).astype(np.float32))
    conv = Conv2d(in_channels, out_channels,
                  kernel_size=1, stride=stride, padding=0, weight_init=weight, has_bias=False,
                  pad_mode="same")
    conv.conv2d.shard(strategy_weight)
    return conv


def conv7x7(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    weight_shape = (out_channels, in_channels, 7, 7)
    weight = Tensor(np.ones(weight_shape).astype(np.float32))
    conv = Conv2d(in_channels, out_channels,
                  kernel_size=7, stride=stride, padding=0, weight_init=weight, has_bias=False,
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


def bn_with_initialize_last(out_channels):
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    gamma = weight_variable_0(shape)
    bn = BatchNorm2d(out_channels, momentum=0.1, eps=0.0001, gamma_init=gamma,
                     beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    bn.bn_train.shard(strategy_bn)
    return bn


def fc_with_initialize(input_channels, out_channels):
    weight_shape = (out_channels, input_channels)
    bias_shape = (out_channels)
    weight = weight_variable_0(weight_shape)
    bias = weight_variable_0(bias_shape)

    return DenseWrap(input_channels, out_channels, weight, bias, has_bias=True,
                     matmul_strategy=strategy_fc_weight_nobias, shard=strategy_tensor_add)


class ResidualBlock(Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1)
        self.bn3 = bn_with_initialize_last(out_channels)

        self.relu1 = P.ReLU().shard(strategy_no_weight)
        self.relu2 = P.ReLU().shard(strategy_no_weight)
        self.relu3 = P.ReLU().shard(strategy_no_weight)
        self.add = Add().shard(strategy_add)

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.add(out, identity)
        out = self.relu3(out)

        return out


class ResidualBlockWithDown(Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlockWithDown, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1)
        self.bn3 = bn_with_initialize_last(out_channels)

        self.relu1 = P.ReLU().shard(strategy_no_weight)
        self.relu2 = P.ReLU().shard(strategy_no_weight)
        self.relu3 = P.ReLU().shard(strategy_no_weight)
        self.down_sample = down_sample

        self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride)
        self.bn_down_sample = bn_with_initialize(out_channels)
        self.add = Add().shard(strategy_add)

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.conv_down_sample(identity)
        identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu3(out)

        return out


class MakeLayer0(Cell):

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer0, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=1, down_sample=True)
        self.b = block(out_channels, out_channels, stride=stride)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x


class ResNet(Cell):

    def __init__(self, block, num_classes=100):
        super(ResNet, self).__init__()
        self.conv1 = conv7x7(3, 64, stride=2)
        self.bn1 = bn_with_initialize(64)
        self.relu = P.ReLU().shard(strategy_no_weight)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = MakeLayer0(
            block, in_channels=64, out_channels=256, stride=1)
        self.pool = M.ReduceMean(keep_dims=True).shard(strategy_no_weight)
        self.fc = fc_with_initialize(64 * block.expansion, num_classes)
        self.flatten = Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.pool(x, (-2, -1))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ResNetModelParallel(Cell):
    def __init__(self, block, num_classes=100):
        super(ResNetModelParallel, self).__init__()
        self.relu = P.ReLU().shard(((1, dev_num, 1, 1),))
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = MakeLayer0(
            block, in_channels=64, out_channels=256, stride=1)
        self.pool = M.ReduceMean(keep_dims=True).shard(strategy_no_weight)
        self.fc = fc_with_initialize(64 * block.expansion, num_classes)
        self.flatten = Flatten()

    def construct(self, x):
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.pool(x, (-2, -1))
        x = self.flatten(x)
        x = self.fc(x)
        return x


def resnet_operator_net(num_classes):
    return ResNet(ResidualBlock, num_classes)


def resnet_model_parallel_net(num_classes):
    return ResNetModelParallel(ResidualBlock, num_classes)


def test_resnet_operator_batch_parallel():
    num_classes = 1024
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=dev_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=dev_num)
    context.set_context(mode=context.GRAPH_MODE)
    predict = Tensor(np.ones([batch_size, 3, 224, 224]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)

    dataset = DatasetLenet(predict, label, 2)
    net = resnet_operator_net(num_classes)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss.softmax_cross_entropy.shard(((dev_num, 1), (dev_num, 1)))
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)

    model = Model(net, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


def test_resnet_model_parallel():
    num_classes = 1024
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=dev_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=dev_num)
    context.set_context(mode=context.GRAPH_MODE)
    predict = Tensor(np.ones([batch_size, 64, 112, 112]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)

    dataset = DatasetLenet(predict, label, 2)
    net = resnet_model_parallel_net(num_classes)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss.softmax_cross_entropy.shard(((dev_num, 1), (dev_num, 1)))
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)

    model = Model(net, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


if __name__ == '__main__':
    test_resnet_model_parallel()
