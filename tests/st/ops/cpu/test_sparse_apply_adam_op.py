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
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

beta1_power = 0.9
beta2_power = 0.999
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sparse_apply_adam = P.SparseApplyAdam()
        self.var = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="m")
        self.v = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="v")

    def construct(self, grad, indices):
        out = self.sparse_apply_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
                                     grad, indices)
        return out


def test_net():
    gradient = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    sparse_apply_adam = Net()
    output = sparse_apply_adam(gradient, indices)
    print(output[0].asnumpy())
