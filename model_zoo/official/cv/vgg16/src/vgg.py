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
"""
Image classifiation.
"""
import math
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
from .utils.var_init import default_recurisive_init, KaimingNormal


def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight = 'ones'
            if args.initialize_mode == "XavierUniform":
                weight_shape = (v, in_channels, 3, 3)
                weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32)

            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias,
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        num_classes (int): Class numbers. Default: 1000.
        batch_norm (bool): Whether to do the batchnorm. Default: False.
        batch_size (int): Batch size. Default: 1.
        include_top(bool): Whether to include the 3 fully-connected layers at the top of the network. Default: True.

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        >>>     num_classes=1000, batch_norm=False, batch_size=1)
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train",
                 include_top=True):
        super(Vgg, self).__init__()
        _ = batch_size
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.include_top = include_top
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])
        if args.initialize_mode == "KaimingNormal":
            default_recurisive_init(self)
            self.custom_init_weight()

    def construct(self, x):
        x = self.layers(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.classifier(x)
        return x

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(num_classes=1000, args=None, phase="train", **kwargs):
    """
    Get Vgg16 neural network with batch normalization.

    Args:
        num_classes (int): Class numbers. Default: 1000.
        args(namespace): param for net init.
        phase(str): train or test mode.

    Returns:
        Cell, cell instance of Vgg16 neural network with batch normalization.

    Examples:
        >>> vgg16(num_classes=1000, args=args, **kwargs)
    """

    if args is None:
        from .config import cifar_cfg
        args = cifar_cfg
    net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase, **kwargs)
    return net
