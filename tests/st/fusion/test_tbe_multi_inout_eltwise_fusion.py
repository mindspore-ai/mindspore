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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_id=4, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.cast = P.Cast()
        self.relu = P.ReLU()
        self.biasadd = P.BiasAdd()

    def construct(self, x, y, k, h):
        z = self.add(x, y)
        z = self.relu(z)
        z = self.relu(z)
        z1 = self.biasadd(z, k)
        z2 = self.biasadd(z, h)
        z = self.add(z1, z2)
        return z


def test_net():
    x = np.random.randn(32, 10).astype(np.float32)
    y = np.random.randn(32, 10).astype(np.float32)
    k = np.random.randn(10).astype(np.float32)
    h = np.random.randn(10).astype(np.float32)
    relu_relu = Net()
    output = relu_relu(Tensor(x), Tensor(y), Tensor(k), Tensor(h))
    print(output.asnumpy())
