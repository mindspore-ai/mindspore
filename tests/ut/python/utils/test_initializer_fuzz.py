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
""" test_initializer_fuzz """
import pytest

import mindspore.nn as nn
from mindspore.train import Model


class Net(nn.Cell):
    """ Net definition """

    def __init__(self, in_str):
        a, b, c, d, e, f, g, h = in_str.strip().split()
        a = int(a)
        b = int(b)
        c = int(b)
        d = int(b)
        e = int(b)
        f = int(b)
        g = int(b)
        h = int(b)

        super(Net, self).__init__()
        self.conv = nn.Conv2d(a, b, c, pad_mode="valid")
        self.bn = nn.BatchNorm2d(d)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(e * f * g, h)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


class LeNet5(nn.Cell):
    """ LeNet5 definition """

    def __init__(self, in_str):
        super(LeNet5, self).__init__()

        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 = in_str.strip().split()
        a1 = int(a1)
        a2 = int(a2)
        a3 = int(a3)
        a4 = int(a4)
        a5 = int(a5)
        a6 = int(a6)
        a7 = int(a7)
        a8 = int(a8)
        a9 = int(a9)
        a10 = int(a10)
        a11 = int(a11)
        a12 = int(a12)
        a13 = int(a13)
        a14 = int(a14)
        a15 = int(a15)

        self.conv1 = nn.Conv2d(a1, a2, a3, pad_mode="valid")
        self.conv2 = nn.Conv2d(a4, a5, a6, pad_mode="valid")
        self.fc1 = nn.Dense(a7 * a8 * a9, a10)
        self.fc2 = nn.Dense(a11, a12)
        self.fc3 = nn.Dense(a13, a14)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=a15)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_shape_error():
    """ for fuzz test"""
    in_str = "3 6 5 6 -6 5 16 5 5 120 120 84 84 3 2"
    with pytest.raises(ValueError):
        net = LeNet5(in_str)  # neural network
        Model(net)
