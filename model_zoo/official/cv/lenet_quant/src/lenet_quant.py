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
"""Manual construct network for LeNet"""

import mindspore.nn as nn
from mindspore.compression.quant import create_quant_config
from mindspore.compression.common import QuantDtype

class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """

    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.qconfig = create_quant_config(per_channel=(True, False), symmetric=(True, False))

        self.conv1 = nn.Conv2dQuant(channel, 6, 5, pad_mode='valid', quant_config=self.qconfig,
                                    quant_dtype=QuantDtype.INT8)
        self.conv2 = nn.Conv2dQuant(6, 16, 5, pad_mode='valid', quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.fc1 = nn.DenseQuant(16 * 5 * 5, 120, quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.fc2 = nn.DenseQuant(120, 84, quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.fc3 = nn.DenseQuant(84, self.num_class, quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)

        self.relu = nn.ActQuant(nn.ReLU(), quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
