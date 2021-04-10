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

"""VGG16 for deeptext"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P

def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    weights = 'ones'
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         pad_mode=pad_mode, weight_init=weights, has_bias=False)]
    layers += [nn.BatchNorm2d(out_channels)]
    return nn.SequentialCell(layers)


class VGG16FeatureExtraction(nn.Cell):
    """VGG16FeatureExtraction for deeptext"""

    def __init__(self):
        super(VGG16FeatureExtraction, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1_1 = _conv(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = _conv(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = _conv(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = _conv(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = _conv(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = _conv(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = _conv(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = _conv(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.cast = P.Cast()

    def construct(self, x):
        """ Construction of VGG """
        x = self.cast(x, mstype.float32)
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        f1 = x

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        f2 = x

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        f3 = x

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        f4 = x

        x = self.max_pool(x)
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        f5 = x

        return f1, f2, f3, f4, f5
