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
model layers
"""
import mindspore.nn as nn
import mindspore.ops as ops


class Conv(nn.Cell):
    """
    conv 2d
    """

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(
            in_channels=inp_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=(kernel_size - 1) // 2,
            has_bias=True,
        )

    def construct(self, x):
        """
        forward
        """

        x = self.conv(x)
        return x


class ConvBNReLU(nn.Cell):
    """
    conv 2d with batch normalize and relu
    """

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1):
        super(ConvBNReLU, self).__init__()
        self.inp_dim = inp_dim
        self.conv = Conv(inp_dim, out_dim, kernel_size, stride)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_dim, momentum=0.9)

    def construct(self, x):
        """
        forward
        """

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Residual(nn.Cell):
    """
    residual block
    """

    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=inp_dim, momentum=0.9)
        self.conv1 = Conv(inp_dim, out_dim // 2, 1)
        self.bn2 = nn.BatchNorm2d(momentum=0.9, num_features=out_dim // 2)
        self.conv2 = Conv(out_dim // 2, out_dim // 2, 3)
        self.bn3 = nn.BatchNorm2d(momentum=0.9, num_features=out_dim // 2)
        self.conv3 = Conv(out_dim // 2, out_dim, 1)
        self.skip_layer = Conv(inp_dim, out_dim, 1)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def construct(self, x):
        """
        forward
        """
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Cell):
    """
    hourglass module
    """

    def __init__(self, n, f):
        super(Hourglass, self).__init__()
        self.up1 = Residual(f, f)
        # Down sampling
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, f)
        self.n = n
        # Use Hourglass recursively
        if self.n > 1:
            self.low2 = Hourglass(n - 1, f)
        else:
            self.low2 = Residual(f, f)
        self.low3 = Residual(f, f)

        # Set upsample size
        sz = [0, 8, 16, 32, 64]
        self.up2 = ops.ResizeNearestNeighbor((sz[n], sz[n]))

    def construct(self, x):
        """
        forward
        """

        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2
