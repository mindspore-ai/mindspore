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
""" tests for quant """
import numpy as np
from mobilenetv2_combined import MobileNetV2

import mindspore.context as context
from mindspore import Tensor
from mindspore import nn
from mindspore.nn.layer import combined
from mindspore.train.quant import quant as qat

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """

    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = combined.Conv2d(
            1, 6, kernel_size=5, batchnorm=True, activation='relu6')
        self.conv2 = combined.Conv2d(6, 16, kernel_size=5, activation='relu')
        self.fc1 = combined.Dense(16 * 5 * 5, 120, activation='relu')
        self.fc2 = combined.Dense(120, 84, activation='relu')
        self.fc3 = combined.Dense(84, self.num_class)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flattern = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.max_pool2d(x)
        x = self.flattern(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
