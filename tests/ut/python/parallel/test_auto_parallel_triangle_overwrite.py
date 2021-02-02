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
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from mindspore import context, Tensor, Parameter
from tests.ut.python.ops.test_math_ops import VirtualLoss

grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)

class GradWarp(nn.Cell):
    def __init__(self, network):
        super(GradWarp, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)

def test_triangle_strategy_consistency():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.mul1 = P.Mul().shard(((2, 4), (2, 4)))
            self.mul2 = P.Mul()
            self.ba1 = P.BiasAdd()
            self.weight = Parameter(Tensor(np.ones([128, 1000]), dtype=ms.float32), name="weight")
            self.bias = Parameter(Tensor(np.ones([1000]), dtype=ms.float32), name="bias")
            self.add = P.Add().shard(((1, 8), (1, 8)))
            self.relu = P.ReLU()

        def construct(self, x):
            out = self.mul1(x, self.weight)
            mul_out = self.mul2(out, self.weight)
            ba_out = self.ba1(out, self.bias)
            ta_out = self.add(mul_out, ba_out)
            out = self.relu(ta_out)
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 1000]), dtype=ms.float32)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()

    net.set_train()
    _executor.compile(net, x, phase='train')
