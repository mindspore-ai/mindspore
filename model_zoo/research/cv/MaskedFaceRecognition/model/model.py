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
"""ResNet."""
import math
import numpy as np
import mindspore
from mindspore import ParameterTuple
import mindspore.nn as nn
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits, L1Loss
from  mindspore.nn import Momentum
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.initializer import HeNormal
from mindspore.common.initializer import Normal
from mindspore  import Tensor
from .stn import STN


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    n = 3*3*out_channel
    normal = Normal(math.sqrt(2. / n))
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=normal)


def _conv1x1(in_channel, out_channel, stride=1):
    n = 1*1*out_channel
    normal = Normal(math.sqrt(2. / n))
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=normal)


def _conv7x7(in_channel, out_channel, stride=1):
    n = 7*7*out_channel
    normal = Normal(math.sqrt(2. / n))
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=normal)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1, use_batch_statistics=None)

def _bn1(channel):
    return nn.BatchNorm1d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1, use_batch_statistics=None)

def _bn1_kaiming(channel):
    return nn.BatchNorm1d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1, use_batch_statistics=None)

def _bn2_kaiming(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1, use_batch_statistics=None)

def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    he_normal = HeNormal()
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=he_normal, bias_init='zeros')


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn(out_channel)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel)])
        self.add = P.TensorAdd()


    def construct(self, x):
        '''construct'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class HardAttn(nn.Cell):
    '''LPD module'''
    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = _fc(128*128, 32)
        self.bn1 = _bn1(32)
        self.fc2 = _fc(32, 4)
        self.bn2 = _bn1(4)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.reduce_mean = P.ReduceMean()


    def construct(self, x):
        '''construct'''
        x = self.reduce_mean(x, 1)
        x_size = self.shape(x)
        x = self.reshape(x, (x_size[0], 128*128))
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.reshape(x, (x_size[0], 4))
        return x


class ResNet(nn.Cell):
    """
    ResNet architecture.
    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 channels,
                 out_channels,
                 strides,
                 num_classes, is_train):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.ha3 = HardAttn(2048)
        self.is_train = is_train
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       channel=channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       channel=channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       channel=channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       channel=channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.max = P.ReduceMax(keep_dims=True)
        self.flatten = nn.Flatten()
        self.global_bn = _bn2_kaiming(out_channels[3])
        self.partial_bn = _bn2_kaiming(out_channels[3])
        normal = Normal(0.001)
        self.global_fc = nn.Dense(out_channels[3], num_classes, has_bias=False, weight_init=normal, bias_init='zeros')
        self.partial_fc = nn.Dense(out_channels[3], num_classes, has_bias=False, weight_init=normal, bias_init='zeros')
        self.theta_0 = Tensor(np.zeros((128, 4)), mindspore.float32)
        self.theta_6 = Tensor(np.zeros((128, 4))+0.6, mindspore.float32)
        self.STN = STN(128, 128)
        self.concat = P.Concat(axis=1)
        self.shape = P.Shape()
        self.tanh = P.Tanh()
        self.slice = P.Slice()
        self.split = P.Split(1, 4)


    def _make_layer(self, block, layer_num, in_channel, channel, out_channel, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []
        resnet_block = block(in_channel, channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)


    def stn(self, x, stn_theta):
        '''stn'''
        x_size = self.shape(x)
        theta = self.tanh(stn_theta)
        theta1, theta5, theta6, theta3 = self.split(theta)
        theta_0 = self.slice(self.theta_0, (0, 0), (x_size[0], 4))
        theta2, theta4, _, _ = self.split(theta_0)
        theta = self.concat((theta1, theta2, theta3, theta4, theta5, theta6))
        flip_feature = self.STN(x, theta)
        return flip_feature, theta5


    def construct(self, x):
        '''construct'''
        stn_theta = self.ha3(x)
        x_p, theta = self.stn(x, stn_theta)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.max(c5, (2, 3))
        out = self.global_bn(out)
        global_f = self.flatten(out)

        x_p = self.conv1(x_p)
        x_p = self.bn1(x_p)
        x_p = self.relu(x_p)
        c1_p = self.maxpool(x_p)

        c2_p = self.layer1(c1_p)
        c3_p = self.layer2(c2_p)
        c4_p = self.layer3(c3_p)
        c5_p = self.layer4(c4_p)

        out_p = self.max(c5_p, (2, 3))
        out_p = self.partial_bn(out_p)
        partial_f = self.flatten(out_p)

        global_out = self.global_fc(global_f)
        partial_out = self.partial_fc(partial_f)
        return global_f, partial_f, global_out, partial_out, theta


class NetWithLossClass(nn.Cell):
    '''net with loss'''
    def __init__(self, network, is_train=True):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.l1_loss = L1Loss()
        self.network = network
        self.is_train = is_train
        self.concat = P.Concat(axis=1)


    def construct(self, x, label1, label2):
        '''construct'''
        global_f, partial_f, global_out, partial_out, theta = self.network(x)
        if not self.is_train:
            out = self.concat((global_f, partial_f))
            return out
        loss_global = self.loss(global_out, label1)
        loss_partial = self.loss(partial_out, label1)
        loss_theta = self.l1_loss(theta, label2)
        loss = loss_global + loss_partial + loss_theta
        return loss


class TrainStepWrap(nn.Cell):
    '''train step wrap'''
    def __init__(self, network, lr, momentum, is_train=True):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = Momentum(self.weights, lr, momentum)
        self.grad = C.GradOperation(get_by_list=True)
        self.is_train = is_train


    def construct(self, x, labels1, labels2):
        '''construct'''
        weights = self.weights
        loss = self.network(x, labels1, labels2)
        if not self.is_train:
            return loss
        grads = self.grad(self.network, weights)(x, labels1, labels2)
        return F.depend(loss, self.optimizer(grads))


class TestStepWrap(nn.Cell):
    """
    Predict method
    """
    def __init__(self, network):
        super(TestStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()


    def construct(self, x, labels):
        '''construct'''
        logits_global, _, _, _, = self.network(x)
        pred_probs = self.sigmoid(logits_global)

        return logits_global, pred_probs, labels


def resnet50(class_num=10, is_train=True):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50(10)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [64, 128, 256, 512],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 1],
                  class_num, is_train)

def resnet101(class_num=1001):
    """
    Get ResNet101 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet101 neural network.

    Examples:
        >>> net = resnet101(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)
