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

import numpy as np
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore import nn, jit
from mindspore.ops import composite as C

class Net(nn.Cell):
    def __init__(self, num_layer):
        super().__init__()
        self.layers = nn.CellList()
        self.dense = nn.Dense(4, 4)
        for _ in range(num_layer):
            self.layers.append(nn.ReLU())
        self.flatten = nn.Flatten()

    @jit(backend="ms_backend")
    def construct(self, x):
        out = x
        out = self.dense(x)
        for layer in self.layers:
            out = layer(out)
        out = self.flatten(out)
        return out

class Grad(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

net = Net(100)
grad_net = Grad(net)
d = Tensor(shape=[None, None], dtype=mstype.float32)
grad_net.set_inputs(d)

input_X = Tensor(np.random.randn(4, 4).astype(np.float32))
ggrad_net = Grad(grad_net)
res = ggrad_net(input_X)
print("AAA", res, "BBB")
print("AAA", res[0].asnumpy().shape, "BBB")
