# Copyright 2025 Huawei Technologies Co., Ltd
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
""" code for analyze fail test """
import numpy as np
from mindspore import Tensor, nn, jit
import mindspore.ops.operations as op


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()
        self.relu = nn.ReLU()

    @jit
    def construct(self, input_x, input_y):
        input_z = self.addn((input_x, input_y))
        out = self.relu(input_z)
        return out


input_1 = Tensor(np.random.randn(2, 3).astype(np.float32))
input_2 = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
net = Net()
net(input_1, input_2)
