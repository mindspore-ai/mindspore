# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _executor
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import _VirtualDataset


class VirtualDatasetNet(nn.Cell):
    def __init__(self):
        super(VirtualDatasetNet, self).__init__()
        self.virtual_dataset = _VirtualDataset()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        self.gelu = P.GeLU()

    def construct(self, x, y, z):
        x, y, z = self.virtual_dataset(x, y, z)
        out = self.gelu(self.matmul1(x, y))
        out = self.matmul2(out, z)
        return out


def test_virtual_dataset():
    x = Tensor(np.ones([128, 32], dtype=np.float32))
    y = Tensor(np.ones([32, 64], dtype=np.float32))
    z = Tensor(np.ones([64, 64], dtype=np.float32))
    network = VirtualDatasetNet()
    _executor.compile(network, x, y, z)
