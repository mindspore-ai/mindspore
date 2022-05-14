# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class MulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, y):
        x = self.mul(x, y)
        return x


class SubNet1(nn.Cell):
    def __init__(self, in_channels=3, out_channels=12):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init="ones")
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = P.ReLU()
        self.neg = P.Neg()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x1 = self.relu(x)
        x2 = self.neg(x)
        return x1, x2


class SubNet2(nn.Cell):
    def __init__(self, in_channels=12, out_channels=12):
        super().__init__()
        self.dense = nn.Dense(in_channels, out_channels, weight_init="ones")
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = P.ReLU()
        self.mul = P.Mul()
        self.split = P.Split(axis=1, output_num=3)
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, x, y):
        x = self.dense(x)
        x = self.bn(x)
        y, _, _ = self.split(y)
        y = self.mean(y, (2, 3))
        x = self.mul(x, y)
        x = self.relu(x)
        return x


class SubNet3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = MulNet()
        self.add = P.Add()
        self.neg = P.Neg()

    def construct(self, x, y):
        z = self.mul(x, y)
        z_1 = self.add(z, y)
        z_2 = self.neg(z)
        return z_1, z_2


class SubNet4(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.neg = P.Neg()

    def construct(self, x, y):
        x = self.add(x, y)
        x = self.neg(x)
        return x


class CompNet(nn.Cell):
    def __init__(self, mul_size, add_size):
        super().__init__()
        mul_np = np.full(mul_size, 0.1, dtype=np.float32)
        add_np = np.full(add_size, 0.1, dtype=np.float32)
        self.mul_weight = mindspore.Parameter(Tensor(mul_np), name="mul_weight")
        self.add_weight = mindspore.Parameter(Tensor(add_np), name="add_weight")
        self.mul = P.Mul()
        self.add = P.Add()
        self.relu = P.ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.split = P.Split(axis=1, output_num=3)
        self.sub_net_1 = SubNet1()
        self.sub_net_2 = SubNet2()
        self.sub_net_3 = SubNet3()
        self.sub_net_4 = SubNet4()

    def construct(self, inputs):
        x, y = self.sub_net_3(inputs, self.mul_weight)
        x_1, x_2 = self.sub_net_1(x)
        x_1 = self.mean(x_1, (2, 3))
        y_1 = self.add(y, self.add_weight)
        x_3 = self.sub_net_2(x_1, self.add_weight)
        y_1, _, _ = self.split(y_1)
        y_2 = self.add(x_2, y_1)
        y_2 = self.sub_net_4(x_2, y_2)
        y_2 = self.mean(y_2, (2, 3))
        z = self.mul(x_3, y_2)
        z_1 = self.relu(z)
        return z_1
