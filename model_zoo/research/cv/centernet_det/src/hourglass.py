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
hourglass backbone
"""
import mindspore.nn as nn

BN_MOMENTUM = 0.9


class Convolution(nn.Cell):
    """
    Convolution block for hourglass.

    Args:
        cin(int): Input channel.
        cout(int): Output channel.
        ks (int): Input kernel size.
        stride(int): Covolution stride. Default: 1.
        with_bn(bool): Specifies whether the layer uses a bias vector. Default: True.
        bias_init(str): Initializer for the bias vector. Default: ‘zeros’.

    Returns:
        Tensor, the feature after covolution.
    """
    def __init__(self, cin, cout, ks, stride=1, with_bn=True, bias_init='zero'):
        super(Convolution, self).__init__()
        pad = (ks - 1) // 2
        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, pad_mode='pad', padding=pad, stride=stride,
                              has_bias=not with_bn, bias_init=bias_init)
        self.bn = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM) if with_bn else nn.SequentialCell()
        self.relu = nn.ReLU()

    def construct(self, x):
        """Defines the computation performed."""
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class Residual(nn.Cell):
    """
    Residual block for hourglass.

    Args:
        cin(int): Input channel.
        cout(int): Output channel.
        ks(int): Input kernel size.
        stride(int): Covolution stride. Default: 1.
        with_bn(bool): Specifies whether the layer uses a bias vector. Default: True.

    Returns:
        Tensor, the feature after covolution.
    """
    def __init__(self, cin, cout, ks, stride=1, with_bn=True):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, pad_mode='pad', padding=1, stride=stride, has_bias=False)
        self.bn1 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)

        self.skip = nn.SequentialCell(
            nn.Conv2d(cin, cout, kernel_size=1, pad_mode='pad', stride=stride, has_bias=False),
            nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        ) if stride != 1 or cin != cout else nn.SequentialCell()
        self.relu = nn.ReLU()

    def construct(self, x):
        """Defines the computation performed."""
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def make_layer(cin, cout, ks, modules, **kwargs):
    layers = [Residual(cin, cout, ks, **kwargs)]
    for _ in range(modules - 1):
        layers.append(Residual(cout, cout, ks, **kwargs))
    return nn.SequentialCell(*layers)


def make_hg_layer(cin, cout, ks, modules, **kwargs):
    layers = [Residual(cin, cout, ks, stride=2)]
    for _ in range(modules - 1):
        layers += [Residual(cout, cout, ks)]
    return nn.SequentialCell(*layers)


def make_layer_revr(cin, cout, ks, modules, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(Residual(cin, cin, ks, **kwargs))
    layers.append(Residual(cin, cout, ks, **kwargs))
    return nn.SequentialCell(*layers)


class Kp_module(nn.Cell):
    """
    The hourglass backbone network.

    Args:
        n(int): The number of stacked hourglass modules.
        dims(array): Residual network input and output dimensions.
        modules(array): The number of stacked residual networks.

    Returns:
        Tensor, the feature map extracted by hourglass network.
    """
    def __init__(self, n, dims, modules, **kwargs):
        super(Kp_module, self).__init__()
        self.n = n
        curr_mod = modules[0]
        next_mod = modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]
        self.up1 = make_layer(
            curr_dim, curr_dim, 3, curr_mod, **kwargs
        )

        self.low1 = make_hg_layer(
            curr_dim, next_dim, 3, curr_mod, **kwargs
        )

        if self.n > 1:
            self.low2 = Kp_module(
                n - 1, dims[1:], modules[1:], **kwargs
            )
        else:
            self.low2 = make_layer(
                next_dim, next_dim, 3, next_mod, **kwargs
            )

        self.low3 = make_layer_revr(
            next_dim, curr_dim, 3, curr_mod, **kwargs
        )

        self.up2 = nn.ResizeBilinear()

    def construct(self, x):
        """Defines the computation performed."""
        up1 = self.up1(x)
        low1 = self.low1(up1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3, scale_factor=2)
        outputs = up1 + up2
        return outputs
