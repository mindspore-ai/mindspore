# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor

import mindspore.hypercomplex.dual as ops
from mindspore.hypercomplex.utils import get_x_and_y, to_2channel


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    else:
        weight_shape = (out_channel, in_channel, 3, 3)
        weight = Tensor(kaiming_normal((2, *weight_shape), mode="fan_out", nonlinearity='relu'))
    if res_base:
        return ops.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                          padding=1, pad_mode='pad', weight_init=weight)
    return ops.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                      padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    else:
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = Tensor(kaiming_normal((2, *weight_shape), mode="fan_out", nonlinearity='relu'))
    if res_base:
        return ops.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                          padding=0, pad_mode='pad', weight_init=weight)
    return ops.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                      padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    else:
        weight_shape = (out_channel, in_channel, 7, 7)
        weight = Tensor(kaiming_normal((2, *weight_shape), mode="fan_out", nonlinearity='relu'))
    if res_base:
        return ops.Conv2d(in_channel, out_channel,
                          kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)
    return ops.Conv2d(in_channel, out_channel,
                      kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel, res_base=False):
    if res_base:
        return ops.BatchNorm2d(channel, eps=1e-5, momentum=0.1,
                               gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return ops.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                           gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, use_se=False):
    if use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
    else:
        weight_shape = (out_channel, in_channel)
        weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
        self.bn1 = _bn(channel)
        if self.use_se and self.stride != 1:
            self.e2 = nn.SequentialCell([_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                         ops.ReLU(), ops.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')])
        else:
            self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
            self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        self.bn3 = _bn(out_channel)
        if self.se_block:
            self.se_global_pool = P.ReduceMean(keep_dims=False)
            self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
            self.se_mul = P.Mul()
        self.relu = ops.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            if self.use_se:
                if stride == 1:
                    self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel,
                                                                         stride, use_se=self.use_se), _bn(out_channel)])
                else:
                    self.down_sample_layer = nn.SequentialCell([ops.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
                                                                _conv1x1(in_channel, out_channel, 1,
                                                                         use_se=self.use_se), _bn(out_channel)])
            else:
                self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                     use_se=self.use_se), _bn(out_channel)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_se and self.stride != 1:
            out = self.e2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se_block:
            out_se = out
            out = self.se_global_pool(out, (2, 3))
            out = self.se_dense_0(out)
            out = self.relu(out)
            out = self.se_dense_1(out)
            out = self.se_sigmoid(out)
            out = F.reshape(out, F.shape(out) + (1, 1))
            out = self.se_mul(out, out_se)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResidualBlockBase(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        res_base (bool): Enable parameter setting of resnet18. Default: True.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlockBase(3, 256, stride=2)
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False,
                 se_block=False,
                 res_base=True):
        super(ResidualBlockBase, self).__init__()
        self.res_base = res_base
        self.conv1 = _conv3x3(in_channel, out_channel, stride=stride, res_base=self.res_base)
        self.bn1d = _bn(out_channel)
        self.conv2 = _conv3x3(out_channel, out_channel, stride=1, res_base=self.res_base)
        self.bn2d = _bn(out_channel)
        self.relu = ops.ReLU()

        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True

        self.down_sample_layer = None
        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                 use_se=use_se, res_base=self.res_base),
                                                        _bn(out_channel, res_base)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1d(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2d(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out


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
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net in layer 3 and layer 4. Default: False.
        res_base (bool): Enable parameter setting of resnet18. Default: False.

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
                 out_channels,
                 strides,
                 num_classes,
                 use_se=False,
                 res_base=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        if self.use_se:
            self.se_block = True

        if self.use_se:
            self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
            self.bn1_0 = _bn(32)
            self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
            self.bn1_1 = _bn(32)
            self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        else:
            self.conv1 = _conv7x7(3, 64, stride=2, res_base=self.res_base)
        self.bn1 = _bn(64, self.res_base)
        self.relu = ops.ReLU()

        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = ops.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        else:
            self.maxpool = ops.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       use_se=self.use_se,
                                       se_block=self.se_block)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       use_se=self.use_se,
                                       se_block=self.se_block)

        self.avgpool = ops.AvgPool2d(4)
        self.concat = P.Concat(1)
        self.flatten = nn.Flatten()
        self.end_point = _fc(16384, num_classes, use_se=self.use_se)

    def construct(self, x):
        x = to_2channel(x[:, :3], x[:, 3:])
        if self.use_se:
            x = self.conv1_0(x)
            x = self.bn1_0(x)
            x = self.relu(x)
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.relu(x)
            x = self.conv1_2(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x_1, x_2 = get_x_and_y(x)
            x_1 = self.pad(x_1)
            x_2 = self.pad(x_2)
            x = to_2channel(x_1, x_2)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.avgpool(x)
        out_x, out_y = get_x_and_y(out)
        out = self.concat([out_x, out_y])
        out = self.flatten(out)
        out = self.end_point(out)
        return out

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        if se_block:
            for _ in range(1, layer_num - 1):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
            layers.append(resnet_block)
        else:
            for _ in range(1, layer_num):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
        return nn.SequentialCell(layers)


def resnet18(class_num=10):
    """
    Get ResNet18 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet18 neural network.

    Examples:
        >>> net = resnet18(10)
    """
    return ResNet(ResidualBlockBase,
                  [2, 2, 2, 2],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  class_num,
                  res_base=True)


def resnet34(class_num=10):
    """
    Get ResNet34 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet34 neural network.

    Examples:
        >>> net = resnet18(10)
    """
    return ResNet(ResidualBlockBase,
                  [3, 4, 6, 3],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  class_num,
                  res_base=True)


def resnet50(class_num=10):
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
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


def se_resnet50(class_num=1001):
    """
    Get SE-ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of SE-ResNet50 neural network.

    Examples:
        >>> net = se-resnet50(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  use_se=True)


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
