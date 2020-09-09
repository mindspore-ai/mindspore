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
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

lr = 0.01
l1 = 0.0
l2 = 0.0
lr_power = -0.5

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fused_sparse_ftrl = P.FusedSparseFtrl(lr=0.1, l1=0.0, l2=0.0, lr_power=-0.5)
        self.var = Parameter(Tensor(np.ones([3, 3]).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.ones([3, 3]).astype(np.float32)), name="accum")
        self.linear = Parameter(Tensor(np.ones([3, 3]).astype(np.float32)), name="linear")

    def construct(self, grad, indices):
        return self.fused_sparse_ftrl(self.var, self.accum, self.linear, grad, indices)

def test_net():
    gradient = Tensor(np.array([-3, 2, 3, 0, 0, 0, -4, -1, -2])
                      .reshape([3, 3]).astype(np.float32))
    indices = Tensor(np.ones([3]), mstype.int32)
    net = Net()
    output = net(gradient, indices)
    print(output)
    print(net.var.data)
    print(net.accum.data)
    print(net.linear.data)
