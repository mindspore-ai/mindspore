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
"""
The model of ResNet50+BAM. MindSpore1.2.0-Ascend.
"""
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from scipy.stats import truncnorm

conv_weight_init = 'HeUniform'


def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, use_se=False):
    if use_se:
        weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    else:
        weight_shape = (out_channel, in_channel, 3, 3)
        weight = _weight_variable(weight_shape)
    return weight


def _conv1x1(in_channel, out_channel, use_se=False):
    if use_se:
        weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    else:
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = _weight_variable(weight_shape)
    return weight


def _conv7x7(in_channel, out_channel, use_se=False):
    if use_se:
        weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    else:
        weight_shape = (out_channel, in_channel, 7, 7)
        weight = _weight_variable(weight_shape)
    return weight


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn1(channel):
    return nn.BatchNorm1d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, use_se=False):
    if use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
    else:
        weight_shape = (out_channel, in_channel)
        weight = _weight_variable(weight_shape)
    return weight


class BasicBlock(nn.Cell):
    """
    BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, pad_mode='pad',
                               padding=1, weight_init=conv_weight_init, has_bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, pad_mode='pad',
                               padding=1, weight_init=conv_weight_init, has_bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def construct(self, x):
        """
        construct
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """
    Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, pad_mode='pad',
                               weight_init=conv_weight_init, has_bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes, momentum=0.9)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, pad_mode='pad',
                               padding=1, weight_init=conv_weight_init, has_bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes, momentum=0.9)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * 4, kernel_size=1, pad_mode='pad',
                               weight_init=conv_weight_init, has_bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=planes * 4, momentum=0.9)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * 4, 16)
        else:
            self.cbam = None

    def construct(self, x):
        """
        construct
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet
    """
    def __init__(self, block, layers, network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                                   weight_init=conv_weight_init, has_bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
            self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7, pad_mode='valid')
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                                   weight_init=conv_weight_init, has_bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.relu = nn.ReLU()

        if att_type == 'BAM':
            self.bam1 = BAM(64 * block.expansion)
            self.bam2 = BAM(128 * block.expansion)
            self.bam3 = BAM(256 * block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Dense(in_channels=512 * block.expansion, out_channels=num_classes,
                           has_bias=True, weight_init=_fc(512 * block.expansion, num_classes), bias_init=0)

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        """
        _make_layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1,
                          stride=stride, pad_mode='pad',
                          weight_init=conv_weight_init, has_bias=False),
                nn.BatchNorm2d(num_features=planes * block.expansion, momentum=0.9),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type == 'CBAM'))
        self.inplanes = planes * block.expansion
        useless_ = 0
        for i in range(1, blocks):
            useless_ += i
            layers.append(block(self.inplanes, planes, use_cbam=att_type == 'CBAM'))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        construct
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = P.AvgPool(4, 4, 'valid')(x)
        x = P.Reshape()(x, (P.Shape()(x)[0], -1,))
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes, att_type):
    """
    ResidualNet
    """
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


# BAM
class Flatten(nn.Cell):
    def construct(self, x):
        return P.Reshape()(x, (P.Shape()(x)[0], -1,))


class ChannelGate(nn.Cell):
    """
    ChannelGate
    """
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c_list = [Flatten()]
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c_list.append(nn.Dense(in_channels=gate_channels[i], out_channels=gate_channels[i + 1],
                                             has_bias=True, weight_init=_fc(gate_channels[i], gate_channels[i + 1]),
                                             bias_init=0))
            self.gate_c_list.append(nn.BatchNorm1d(num_features=gate_channels[i + 1], momentum=0.9))
            self.gate_c_list.append(nn.ReLU())
        self.gate_c_list.append(nn.Dense(in_channels=gate_channels[-2], out_channels=gate_channels[-1],
                                         has_bias=True, weight_init=_fc(gate_channels[-2], gate_channels[-1]),
                                         bias_init=0
                                         ))
        self.gate_c = nn.SequentialCell(self.gate_c_list)

    def construct(self, in_tensor):
        """
        construct
        """
        size = F.shape(in_tensor)
        avg_pool = P.AvgPool(size[2], size[2])(in_tensor)
        expand_dims = P.ExpandDims()
        need = self.gate_c(avg_pool)
        need = expand_dims(need, 2)
        need = expand_dims(need, 3)
        broadcast_to = P.BroadcastTo(size)
        need = broadcast_to(need)
        return need


class SpatialGate(nn.Cell):
    """
    SpatialGate
    """
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s_list = [nn.Conv2d(in_channels=gate_channel, out_channels=gate_channel // reduction_ratio,
                                      weight_init=conv_weight_init,
                                      kernel_size=1, pad_mode='pad', has_bias=True)]
        self.gate_s_list.append(nn.BatchNorm2d(num_features=gate_channel // reduction_ratio, momentum=0.9))
        self.gate_s_list.append(nn.ReLU())
        useless_2 = 0
        for i in range(dilation_conv_num):
            self.gate_s_list.append(nn.Conv2d(in_channels=gate_channel // reduction_ratio,
                                              out_channels=gate_channel // reduction_ratio, kernel_size=3,
                                              pad_mode='pad', padding=dilation_val, dilation=dilation_val,
                                              weight_init=conv_weight_init,
                                              has_bias=True))
            self.gate_s_list.append(nn.BatchNorm2d(num_features=gate_channel // reduction_ratio, momentum=0.9))
            self.gate_s_list.append(nn.ReLU())
            useless_2 += i
        self.gate_s_list.append(nn.Conv2d(in_channels=gate_channel // reduction_ratio, out_channels=1, kernel_size=1,
                                          weight_init=conv_weight_init,
                                          pad_mode='pad', has_bias=True))
        self.gate_s = nn.SequentialCell(self.gate_s_list)

    def construct(self, in_tensor):
        """
        construct
        """
        size = F.shape(in_tensor)
        broadcast_to = P.BroadcastTo(size)
        return broadcast_to(self.gate_s(in_tensor))


class BAM(nn.Cell):
    """
    BAM
    """
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def construct(self, in_tensor):
        """
        construct
        """
        att = 1 + P.Sigmoid()(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor


# CBAM
class BasicConv(nn.Cell):
    """
    BasicConv
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, dilation=dilation, has_bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        """
        construct
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate_CBAM(nn.Cell):
    """
    ChannelGate_CBAM
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate_CBAM, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.SequentialCell([
            Flatten(),
            nn.Dense(in_channels=gate_channels, out_channels=gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Dense(in_channels=gate_channels // reduction_ratio, out_channels=gate_channels)
            ])
        self.reducemax_ = P.ReduceMax(keep_dims=True)

    def construct(self, x):
        """
        construct
        """
        size = F.shape(x)
        avg_pool = P.AvgPool((size[2], size[3]), (size[2], size[3]))(x)
        channel_att_raw = self.mlp(avg_pool)
        channel_att_sum = channel_att_raw
        max_pool = self.reducemax_(x, (2, 3))
        channel_att_raw = self.mlp(max_pool)
        channel_att_sum = channel_att_sum + channel_att_raw

        scale = P.Sigmoid()(channel_att_sum)
        scale = P.ExpandDims()(scale, 2)
        scale = P.ExpandDims()(scale, 3)
        scale = P.BroadcastTo(F.shape(x))(scale)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = P.Reshape()(tensor, (P.Shape()(tensor)[0], P.Shape()(tensor)[1], -1,))
    s = P.ReduceMax(keep_dims=True)(tensor_flatten, 2, keepdims=True)
    outputs = s + P.Exp()(tensor_flatten - s)
    outputs = P.ReduceSum(keep_dims=True)(outputs, 2)
    outputs = P.Log()(outputs)
    return outputs


class ChannelPool(nn.Cell):
    """
    ChannelPool
    """
    def __init__(self):
        super(ChannelPool, self).__init__()
        self.reducemax_ = P.ReduceMax(keep_dims=True)
        self.reducemean_ = P.ReduceMean(keep_dims=True)
        self.concat_ = P.Concat(axis=1)

    def construct(self, x):
        """
        construct
        """
        first_ = self.reducemax_(x, 1)
        second_ = self.reducemean_(x, 1)
        return self.concat_((first_, second_))


class SpatialGate_CBAM(nn.Cell):
    """
    SpatialGate_CBAM
    """
    def __init__(self):
        super(SpatialGate_CBAM, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def construct(self, x):
        """
        construct
        """
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = P.Sigmoid()(x_out)
        return x * scale


class CBAM(nn.Cell):
    """
    CBAM
    """
    def __init__(self, gate_channels, reduction_ratio=16, no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate_CBAM = ChannelGate_CBAM(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate_CBAM = SpatialGate_CBAM()

    def construct(self, x):
        """
        construct
        """
        x_out = self.ChannelGate_CBAM(x)
        if not self.no_spatial:
            x_out = self.SpatialGate_CBAM(x_out)
        return x_out
