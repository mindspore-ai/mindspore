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
"""blocks net."""
import mindspore.nn as nn
import mindspore.ops as ops


class FeatureFusionBlock(nn.Cell):
    """FeatureFusionBlock."""
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.resize_bilinear = ops.ResizeBilinear
        self.shape = ops.Shape()

    def construct(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        size_x = self.shape(output)[2] * 2
        size_y = self.shape(output)[3] * 2
        output = self.resize_bilinear((size_x, size_y))(output)
        return output


class ResidualConvUnit(nn.Cell):
    """ResidualConvUnit."""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, has_bias=True,
            padding=1, pad_mode="pad"
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, has_bias=True,
            padding=1, pad_mode="pad"
        )
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class Interpolate(nn.Cell):
    """Interpolate."""
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.resize_bilinear = ops.ResizeBilinear
        self.scale_factor = scale_factor
        self.shape = ops.Shape()

    def construct(self, x):
        size_x = self.shape(x)[2] * self.scale_factor
        size_y = self.shape(x)[3] * self.scale_factor
        x = self.resize_bilinear((size_x, size_y))(x)
        return x
