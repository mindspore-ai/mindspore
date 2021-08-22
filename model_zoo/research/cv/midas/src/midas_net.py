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
"""net."""
import numpy as np
from mindspore import ops
from mindspore import ParameterTuple
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops.operations import Add, Split, Concat
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from src.custom_op import SEBlock, GroupConv
from src.blocks_ms import Interpolate, FeatureFusionBlock
from src.loss import ScaleAndShiftInvariantLoss

from src.config import config


def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad", group=groups)


def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad", group=groups)


def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad", group=groups)


class _DownSample(nn.Cell):
    """
    Downsample for ResNext-ResNet.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride size for the 1*1 convolutional layer.

    Returns:
        Tensor, output tensor.

    Examples:
        >>>DownSample(32, 64, 2)
    """

    def __init__(self, in_channels, out_channels, stride):
        super(_DownSample, self).__init__()
        self.conv = conv1x1(in_channels, out_channels, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class BasicBlock(nn.Cell):
    """
    ResNet basic block definition.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>>BasicBlock(32, 256, stride=2)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None, use_se=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = P.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)

        self.down_sample_flag = False
        if down_sample is not None:
            self.down_sample = down_sample
            self.down_sample_flag = True

        self.add = Add()

    def construct(self, x):
        """construct."""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.down_sample_flag:
            identity = self.down_sample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    """
    ResNet Bottleneck block definition.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride size for the initial convolutional layer. Default: 1.

    Returns:
        Tensor, the ResNet unit's output.

    Examples:
        >>>Bottleneck(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None,
                 base_width=64, groups=1, use_se=False, **kwargs):
        super(Bottleneck, self).__init__()

        width = int(out_channels * (base_width / 64.0)) * groups
        self.groups = groups
        self.conv1 = conv1x1(in_channels, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = P.ReLU()

        self.conv3x3s = nn.CellList()

        self.conv2 = GroupConv(width, width, 3, stride, pad=1, groups=groups)
        self.op_split = Split(axis=1, output_num=self.groups)
        self.op_concat = Concat(axis=1)

        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels * self.expansion)

        self.down_sample_flag = False
        if down_sample is not None:
            self.down_sample = down_sample
            self.down_sample_flag = True

        self.cast = P.Cast()
        self.add = Add()

    def construct(self, x):
        """construct."""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.se(out)

        if self.down_sample_flag:
            identity = self.down_sample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class MidasNet(nn.Cell):
    """Network for monocular depth estimation.
    """
    def __init__(self, block=Bottleneck, width_per_group=config.width_per_group,
                 groups=config.groups, use_se=False,
                 features=config.features, non_negative=True,
                 expand=False):
        super(MidasNet, self).__init__()
        self.in_channels = config.in_channels
        self.groups = groups
        self.layers = config.layers
        self.base_width = width_per_group
        self.backbone_conv = conv7x7(3, self.in_channels, stride=2, padding=3)
        self.backbone_bn = nn.BatchNorm2d(self.in_channels)
        self.backbone_relu = P.ReLU()
        self.backbone_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.backbone_layer1 = self._make_layer(block, 64, self.layers[0], use_se=use_se)
        self.backbone_layer2 = self._make_layer(block, 128, self.layers[1], stride=2, use_se=use_se)
        self.backbone_layer3 = self._make_layer(block, 256, self.layers[2], stride=2, use_se=use_se)
        self.backbone_layer4 = self._make_layer(block, 512, self.layers[3], stride=2, use_se=use_se)
        self.out_channels = 512 * block.expansion
        out_shape1 = features
        out_shape2 = features
        out_shape3 = features
        out_shape4 = features
        self.non_negative = non_negative
        if expand:
            out_shape1 = features
            out_shape2 = features * 2
            out_shape3 = features * 4
            out_shape4 = features * 8
        self.layer1_rn_scratch = nn.Conv2d(
            256, out_shape1, kernel_size=3, stride=1, has_bias=False,
            padding=1, pad_mode="pad", group=1
        )
        self.layer2_rn_scratch = nn.Conv2d(
            512, out_shape2, kernel_size=3, stride=1, has_bias=False,
            padding=1, pad_mode="pad", group=1
        )
        self.layer3_rn_scratch = nn.Conv2d(
            1024, out_shape3, kernel_size=3, stride=1, has_bias=False,
            padding=1, pad_mode="pad", group=1
        )
        self.layer4_rn_scratch = nn.Conv2d(
            2048, out_shape4, kernel_size=3, stride=1, has_bias=False,
            padding=1, pad_mode="pad", group=1
        )
        self.refinenet4_scratch = FeatureFusionBlock(features)
        self.refinenet3_scratch = FeatureFusionBlock(features)
        self.refinenet2_scratch = FeatureFusionBlock(features)
        self.refinenet1_scratch = FeatureFusionBlock(features)
        self.output_conv_scratch = nn.SequentialCell([
            nn.Conv2d(
                features, 128, kernel_size=3, stride=1, has_bias=True,
                padding=1, pad_mode="pad"
            ),
            Interpolate(scale_factor=2),
            nn.Conv2d(
                128, 32, kernel_size=3, stride=1, has_bias=True,
                padding=1, pad_mode="pad"
            ),
            nn.ReLU(),
            nn.Conv2d(
                32, 1, kernel_size=1, stride=1, has_bias=True,
                padding=0, pad_mode="pad"
            ),
            nn.ReLU() if non_negative else ops.Identity(),
        ])
        self.squeeze = ops.Squeeze(1)

    def construct(self, x):
        """construct pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        x = self.backbone_conv(x)
        x = self.backbone_bn(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)
        layer1 = self.backbone_layer1(x)
        layer2 = self.backbone_layer2(layer1)
        layer3 = self.backbone_layer3(layer2)
        layer4 = self.backbone_layer4(layer3)
        layer_1_rn = self.layer1_rn_scratch(layer1)
        layer_2_rn = self.layer2_rn_scratch(layer2)
        layer_3_rn = self.layer3_rn_scratch(layer3)
        layer_4_rn = self.layer4_rn_scratch(layer4)
        path_4 = self.refinenet4_scratch(layer_4_rn)
        path_3 = self.refinenet3_scratch(path_4, layer_3_rn)
        path_2 = self.refinenet2_scratch(path_3, layer_2_rn)
        path_1 = self.refinenet1_scratch(path_2, layer_1_rn)
        out = self.output_conv_scratch(path_1)
        result = self.squeeze(out)
        return result

    def _make_layer(self, block, out_channels, blocks_num, stride=1, use_se=False):
        """_make_layer"""
        down_sample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            down_sample = _DownSample(self.in_channels,
                                      out_channels * block.expansion,
                                      stride=stride)

        layers = [block(self.in_channels,
                        out_channels,
                        stride=stride,
                        down_sample=down_sample,
                        base_width=self.base_width,
                        groups=self.groups,
                        use_se=use_se)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks_num):
            layers.append(block(self.in_channels, out_channels,
                                base_width=self.base_width, groups=self.groups, use_se=use_se))

        return nn.SequentialCell(layers)

    def get_out_channels(self):
        return self.out_channels


class Loss(nn.Cell):
    def __init__(self):
        super(Loss, self).__init__()
        self.lossvalue = ScaleAndShiftInvariantLoss()

    def construct(self, prediction, mask, target):
        loss_value = self.lossvalue(prediction, mask, target)
        return loss_value


class NetwithCell(nn.Cell):
    """NetwithCell."""

    def __init__(self, net, loss):
        super(NetwithCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, image, mask, depth):
        prediction = self._net(image)
        return self._loss(prediction, mask, depth)

    @property
    def backbone_network(self):
        return self._net


class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        reduce_flag (bool): The reduce flag. Default value is False.
        mean (bool): Allreduce method. Default value is False.
        degree (int): Device number. Default value is None.
    """

    def __init__(self, network, optimizer, sens=1.0, reduce_flag=False, mean=True, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.sens = Tensor((np.ones((1,)) * sens).astype(np.float32))
        self.reduce_flag = reduce_flag
        self.hyper_map = C.HyperMap()
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *ds):
        weights = self.weights
        loss = self.network(*ds)
        grads = self.grad(self.network, weights)(*ds, self.sens)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)

        self.optimizer(grads)
        return loss
